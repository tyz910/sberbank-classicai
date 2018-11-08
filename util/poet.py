from util.nlp import Tokenizer
from gensim.models import KeyedVectors
from util.models import PoetLineClf, PoetLineAligner
import pandas as pd
import numpy as np
import itertools
import pickle
import re


class Poet:
    def __init__(self, tokenizer = None):
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        self.poet_lines = {}
        self.poet_lines_vec = KeyedVectors(300)
        self.lines_model = PoetLineAligner(self)
        self.style_model = PoetLineClf()

    def init(self):
        keys = list(self.poet_lines.keys())
        self.poet_lines_vec.add(
            keys,
            [self.tokenizer.get_tokens_vec(line["tokens"]) for line in self.poet_lines.values()]
        )

        poet_predicts = self.style_model.predict_proba([line["tokens"] for line in self.poet_lines.values()])
        for i, poet_predict in enumerate(poet_predicts):
            key = keys[i]
            for p, proba in poet_predict.items():
                if p not in self.poet_lines[key]["poets"]:
                    self.poet_lines[key]["poets"][p] = proba

    def save(self, path="/data/"):
        with open(path + "poet_lines.pickle", "wb") as f:
            pickle.dump(self.poet_lines, f)

    def load(self, path="/data/"):
        with open(path + "poet_lines.pickle", "rb") as f:
            self.poet_lines = pickle.load(f)
            self.init()

    def clean_line(self, line):
        return re.sub("[()*<>[\]]", "", line)

    def read_json(self, path="/data/classic_poems.json"):
        df = pd.read_json(path)

        for i, row in df.iterrows():
            print(i)
            for line in row["content"].split("\n"):
                if len(line.strip()) > 0:
                    self.add_poet_line(line, row["poet_id"])

    def add_poet_line(self, line, poet=None):
        line = self.clean_line(line)
        if len(line) < 60:
            if line not in self.poet_lines:
                tokens = self.tokenizer.tokenize(line)

                hasWv = False
                hasPhoneme = False
                for t in tokens:
                    if not hasWv and t.wv is not None:
                        hasWv = True

                    if not hasPhoneme and t.phoneme is not None:
                        hasPhoneme = True

                    if hasWv and hasPhoneme:
                        break

                if not hasWv or not hasPhoneme:
                    return

                line_data = {
                    "tokens": tokens,
                    "poets": {}
                }

                if poet is not None:
                    line_data["poets"][poet] = 1.0

                self.poet_lines[line] = line_data

    def get_poet_lines(self, seed_tokens, poet="pushkin", topn=100, exclude_lines=None):
        seed_tokens = [t for t in seed_tokens if t.pos not in {"NUM"}]

        lines = []
        cache = set()

        seed_vec = self.tokenizer.get_tokens_vec(seed_tokens)
        for line, sim in self.poet_lines_vec.most_similar_cosmul(positive=[seed_vec], topn=1000):
            if "плев_о_чки" in line:
                continue

            if exclude_lines is not None:
                if line in exclude_lines:
                    continue

            l = self.poet_lines[line]
            poets = l["poets"]
            if poet in poets:
                cache_key = ",".join(sorted([t.lemma for t in l["tokens"]]))
                if cache_key not in cache:
                    cache.add(cache_key)

                    last_token = None
                    for last_token in reversed(l["tokens"]):
                        if last_token.phoneme is not None:
                            break

                    lines.append({
                        "line": line,
                        "sim": sim,
                        "style": poets[poet],
                        "other": "other" in poets,
                        "last_token": last_token,
                        "lemmas": {t.lemma for t in l["tokens"] if t.wv is not None}
                    })

                    if len(lines) == topn:
                        return lines

        return lines

    def get_rhyme_lines(self, poet_lines):
        lines = []
        for l in itertools.combinations(poet_lines, r=2):
            if l[0]["last_token"].lemma != l[1]["last_token"].lemma:
                if l[0]["last_token"].is_phonetic_match(l[1]["last_token"]):
                    score = (l[0]["sim"] + l[1]["sim"]) / 2
                    score += 0.05 * (l[0]["style"] + l[1]["style"]) / 2
                    if (len(l[0]["last_token"].phoneme) + len(l[1]["last_token"].phoneme)) / 2 >= 3:
                        score += 0.1

                    lines.append({
                        "lines": l,
                        "score": score,
                    })

        return lines

    def get_best_rhyme_lines(self, rhyme_lines, num_blocks=2):
        best_score = 0
        result = None

        for candidate in itertools.combinations(rhyme_lines, r=num_blocks):
            lines = set()
            lemmas = set()
            lemmas_cnt = 0
            other_cnt = 0
            score = np.mean([lpair["score"] for lpair in candidate])

            for lpair in candidate:
                for l in lpair["lines"]:
                    if l["line"] not in lines:
                        lines.add(l["line"])
                    else:
                        score -= 10

                    for lemma in l["lemmas"]:
                        if lemma not in lemmas:
                            lemmas.add(lemma)
                        else:
                            lemmas_cnt += 1

                    if l["other"]:
                        other_cnt += 1

            score -= 0.05 * lemmas_cnt

            if other_cnt > 2:
                score -= 0.5

            if score > best_score:
                best_score = score
                result = candidate

        if result is not None:
            return [[l["line"] for l in lpair["lines"]] for lpair in result]
        else:
            return None

    def get_poem(self, seed, poet="pushkin", exclude_lines=None):
        if isinstance(seed, str):
            seed = self.tokenizer.tokenize(seed)

        num_lines = 100
        while True:
            poet_lines = self.get_poet_lines(seed, poet, num_lines, exclude_lines)
            rlines = self.get_rhyme_lines(poet_lines)
            lines = self.get_best_rhyme_lines(rlines)
            if lines is None:
                num_lines += 100
            else:
                break

        return self.lines_model.align_lines(lines[0] + lines[1])
