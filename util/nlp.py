import html
import pymorphy2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pymystem3 import Mystem
from russtress import Accent
from rnnmorph.predictor import RNNMorphPredictor
from rupo.api import Engine
from gensim.models import KeyedVectors
from phonetic_algorithmRu.transcription import __lev_distance__, phrase_transformer


POS_MAP = {
    "A": "ADJ",
    "ADV": "ADV",
    "ADVPRO": "ADV",
    "ANUM": "ADJ",
    "APRO": "DET",
    "COM": "ADJ",
    "CONJ": "CONJ",
    "INTJ": "INTJ",
    "NUM": "NUM",
    "PART": "PART",
    "PR": "ADP",
    "S": "NOUN",
    "SPRO": "PRON",
    "V": "VERB",
}


class DummyStressPredictor:
    def predict(self, text: str) -> List[int]:
        return []


class Token:
    def __init__(self, text):
        self.text = text
        self.lemma = text
        self.before = ""
        self.after = ""
        self.pos = None
        self.stress_pos = None
        self.stress_syl = None
        self.tag = None
        self.tag_vec = None
        self.phoneme = None
        self.syllables = None
        self.wv = None

    def is_phonetic_match(self, token2: 'Token') -> bool:
        if self.phoneme is not None and token2.phoneme is not None:
            max_phonetic_distance = 0.0
            if (len(self.phoneme) + len(token2.phoneme)) / 2 > 2:
                max_phonetic_distance += 0.1

            return __lev_distance__(self.phoneme, token2.phoneme) <= max_phonetic_distance
        else:
            return False

    def __repr__(self) -> str:
        text = self.text
        if self.stress_pos is not None:
            text = text[:self.stress_pos + 1] + "'" + text[self.stress_pos + 1:]
        return "{} <{}_{}>".format(text, self.lemma, self.pos)


class TokenForms:
    def __init__(self, forms: Dict[str, Token], tokenizer: 'Tokenizer'):
        self.forms = forms
        self.tokenizer = tokenizer
        self.best_forms = {}

    def get_best_form(self, token: Token, check_phonetic: bool=False) -> Tuple[Token, float]:
        if token.text not in self.best_forms:
            best = None
            best_score = 0.0

            for f, form in self.forms.items():
                if form.pos == token.pos:
                    if check_phonetic and not form.is_phonetic_match(token):
                        continue

                    score = 4.0 - sum(v1 != v2 for v1, v2 in zip(form.tag_vec, token.tag_vec))

                    if form.stress_syl == token.stress_syl:
                        score += 1.5

                    if form.syllables == token.syllables:
                        score += 1.0

                    if form.wv and token.wv:
                        score += 2.5 * self.tokenizer.wv.cosine_similarities(
                            self.tokenizer.wv[form.wv],
                            [self.tokenizer.wv[token.wv]]
                        )[0]

                    if score > best_score:
                        best_score = score
                        best = form

            self.best_forms[token.text] = (best, best_score)

        return self.best_forms[token.text]

    def __repr__(self) -> str:
        return list(self.forms.keys()).__repr__()


class Tokenizer:
    def __init__(self, wv: str="/data/ruwikiruscorpora_upos_skipgram_300_2_2018.vec.gz"):
        self.mystem = Mystem()
        self.accent = Accent()
        self.rnnmorph = RNNMorphPredictor()
        self.pymorhy2 = pymorphy2.MorphAnalyzer()
        self.rupo = Engine()
        self.rupo.stress_predictors["ru"] = DummyStressPredictor()
        self.wv = KeyedVectors.load_word2vec_format(wv)

    def get_tokens_vec(self, tokens: List[Token]) -> np.ndarray:
        vecs = [self.wv[t.wv] for t in tokens if t.wv is not None]
        return np.mean(vecs, axis=0) if len(vecs) > 0 else np.zeros(300)

    def tokenize(self, text: str) -> List[Token]:
        tokens = self.__mystem_tokenize(html.unescape(text))
        tokens = self.__set_stresses(tokens)
        tokens = self.__set_morph_tags(tokens)
        tokens = self.__set_phonetic(tokens)
        tokens = self.__set_wv(tokens)

        return tokens

    def get_token_forms(self, token: Token) -> Optional[TokenForms]:
        parse = self.pymorhy2.parse(token.lemma)[0]
        is_name = "Name" in parse.tag
        is_title = token.text[0].isupper()

        tokens = []
        for form in parse.lexeme:
            if is_name and "Name" not in form.tag:
                continue

            token = Token(form.word)
            token.lemma = form.normal_form
            if is_title:
                token.text = token.text.title()

            self.__set_stresses([token])
            self.__set_morph_tags([token], set_pos=True)
            tokens.append(token)

        tokens = self.__set_phonetic(tokens)
        tokens = self.__set_wv(tokens)

        return TokenForms({t.text: t for t in tokens}, self) if len(tokens) > 0 else None

    def __set_wv(self, tokens: List[Token]) -> List[Token]:
        for t in tokens:
            if t.pos is not None:
                t.wv = t.lemma + "_" + t.pos
                if t.wv not in self.wv:
                    t.wv = None

        return tokens

    def __set_phonetic(self, tokens: List[Token]) -> List[Token]:
        for i, t in enumerate(tokens):
            if t.pos not in {"CONJ", "ADP"}:
                try:
                    word = self.rupo.get_markup(t.text).lines[0].words[0]
                except IndexError:
                    continue

                try:
                    if t.stress_pos is None:
                        t.stress_pos = word.syllables[0].vowel()

                    word.set_stresses([t.stress_pos])
                except IndexError:
                    pass

                for i, syllable in enumerate(reversed(word.syllables)):
                    if syllable.stress >= 0:
                        try:
                            t.phoneme = phrase_transformer(t.text[syllable.stress:], stresses=[i + 1])[0][0]
                            t.stress_syl = i + 1
                            t.syllables = len([s.text for s in word.syllables])
                        except (ValueError, KeyError, AttributeError):
                            # print("Fail phonetic for: " + t.text)
                            pass

                        break

        return tokens

    def __set_morph_tags(self, tokens: List[Token], set_pos: bool=False) -> List[Token]:
        for i, t in enumerate(self.rnnmorph.predict([t.text for t in tokens])):
            tokens[i].tag = t.tag
            tokens[i].tag_vec = t.vector
            if set_pos:
                tokens[i].pos = t.pos

        return tokens

    def __set_stresses(self, tokens: List[Token]) -> List[Token]:
        text = " ".join([t.text for t in tokens])
        stressed_text = self.accent.put_stress(text, stress_symbol="|!|")

        for i, w in enumerate(stressed_text.split()):
            tokens[i].stress_pos = w.rfind("|!|") - 1
            if tokens[i].stress_pos < 0:
                tokens[i].stress_pos = None

        return tokens

    def __mystem_tokenize(self, text: str) -> List[Token]:
        token = None
        tokens = []
        space = ""

        for t in self.mystem.analyze(" ".join(text.split())):
            if "analysis" in t:
                if token is not None:
                    token.after = space
                    space = ""

                token = Token(t["text"])
                token.before = space
                space = ""

                if len(t["analysis"]) > 0:
                    pos = t["analysis"][0]["gr"].split(',')[0]
                    pos = pos.split('=')[0].strip()
                    token.pos = POS_MAP[pos] if pos in POS_MAP else None
                    token.lemma = t["analysis"][0]["lex"].lower().strip()

                tokens.append(token)
            elif t["text"] != "\n":
                space += t["text"]

        if space != "" and token is not None:
            token.after = space

        return tokens
