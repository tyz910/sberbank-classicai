import numpy as np
import re
from util.poet import Poet


class FinalPoemGenerator:
    def __init__(self):
        print(">>>> LOAD WORD VECTORS. KEEP CALM.")
        self.poet = Poet()

        print(">>>> LOAD POET LINES.")
        self.poet.load()

        print(">>>> WARMUP.")
        self.generate("pushkin", "Дождь за окном")

    def generate(self, style, seed):
        seed = self.poet.tokenizer.tokenize(seed)
        poem_lines = self.poet.get_poem(seed, style)
        lines_tokens = [self.poet.poet_lines[l]["tokens"] for l in poem_lines]
        poem = Poem(self.poet, lines_tokens)
        poem.seed(seed)
        text = self.finalize_text(poem.get_text())
        return text

    def finalize_text(self, text):
        r = "(\w)(\W+)$"
        lines = text.strip("\n").split("\n")
        result = re.search(r, lines[-1])
        if result is None:
            lines[-1] = lines[-1].strip() + "."
        elif result.group(2) not in [".", "!", "?", "…"]:
            lines[-1] = re.sub(r, "\g<1>.", lines[-1])

        return "\n".join(lines)


class PoemToken:
    def __init__(self, token):
        self.token = token
        self.replace = None
        self.in_seed = False
        self.is_last = False

    def __repr__(self):
        rep = self.token.__repr__()

        if self.replace is not None:
            rep += " ==> " + self.replace.__repr__()

        if self.in_seed:
            rep += " <SEED>"

        if self.is_last:
            rep += " <BR>"

        return rep


class Poem:
    def __init__(self, poet, lines_tokens):
        self.poet = poet
        self.tokens = []

        for line in lines_tokens:
            last = None
            for t in line:
                pt = PoemToken(t)
                self.tokens.append(pt)

                if t.phoneme is not None:
                    last = pt

            if last is not None:
                last.is_last = True

    def get_text(self):
        text = ""
        new_line = True

        for t in self.tokens:
            if t.replace is not None:
                word = t.token.before + t.replace.text + t.token.after.strip("\n")
            else:
                word = t.token.before + t.token.text + t.token.after.strip("\n")

            if new_line:
                word = word.title()
                new_line = False

            text += word
            if t.is_last:
                text += "\n"
                new_line = True

        return text

    def seed(self, seed_tokens):
        first_token = seed_tokens[0]
        first_token_text = first_token.text
        parse = self.poet.tokenizer.pymorhy2.parse(first_token_text)[0]
        first_token_text = first_token_text.lower()
        for tag in ["Geox", "Name", "Surn", "Patr"]:
            if tag in parse.tag:
                first_token_text = first_token_text.title()
                break
        first_token.text = first_token_text

        seed_tokens = [t for t in seed_tokens if t.pos in {"NOUN", "VERB", "ADJ"}]

        to_insert = {}
        for seed_token in seed_tokens:
            in_poem = False
            for poem_token in self.tokens:
                if poem_token.replace is None:
                    if poem_token.token.lemma == seed_token.lemma:
                        poem_token.in_seed = True
                        in_poem = True

            if not in_poem and seed_token.lemma not in to_insert:
                to_insert[seed_token.lemma] = seed_token

        if len(to_insert) > 0:
            self.insert_seed_tokens(to_insert)

    def insert_seed_tokens(self, seed_tokens, max_insert=3):
        poem_tokens = [t for t in self.tokens if t.replace is None and not t.in_seed]
        score_matrix = np.zeros((len(seed_tokens), len(poem_tokens)))
        seed_forms = {}

        i = 0
        for s, seed_token in seed_tokens.items():
            f = seed_forms[s] = self.poet.tokenizer.get_token_forms(seed_token)
            if f is not None:
                for j, poem_token in enumerate(poem_tokens):
                    max_phonetic = 0.1 if poem_token.is_last else None
                    form, score = f.get_best_form(poem_token.token, max_phonetic)
                    if score > 0:
                        score_matrix[i, j] = score

            i += 1

        seed_keys = list(seed_tokens.keys())
        for i, j in self.get_best_replace_idx(score_matrix)[:max_insert]:
            poem_token = poem_tokens[j]
            poem_token.replace, score = seed_forms[seed_keys[i]].get_best_form(poem_token.token)

    def get_best_replace_idx(self, m):
        best = []

        while True:
            idx = np.unravel_index(np.argmax(m), m.shape)
            if m[idx] > 0:
                m[idx[0], :] = 0
                m[:, idx[1]] = 0
                best.append(idx)
            else:
                break

        return best

    def __repr__(self):
        return self.tokens.__repr__()
