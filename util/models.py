from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Masking
import keras
import random
import itertools
import numpy as np


class PoetLineClf:
    BATCH_SIZE = 512
    MAX_LINE_LEN = 10
    EMB_SIZE = 54

    POETS = ['pushkin', 'esenin', 'blok', 'tyutchev', 'mayakovskij']
    POET_MAP = {
        'pushkin': 0,
        'esenin': 1,
        'blok': 2,
        'tyutchev': 3,
        'mayakovskij': 4,
    }

    def __init__(self, build_model=False):
        if build_model:
            self.model = self.build_model()
        else:
            self.load()

    def extract_train_lines(self, poet, df):
        class_weight = {}
        lines = []

        for i, row in df.iterrows():
            for line in row["content"].split("\n"):
                line = poet.clean_line(line)
                if line in poet.poet_lines and len(poet.poet_lines[line]["tokens"]) <= self.MAX_LINE_LEN:
                    poet_idx = self.POET_MAP[row["poet_id"]]
                    lines.append((line, poet.poet_lines[line]["tokens"], poet_idx))
                    if poet_idx not in class_weight:
                        class_weight[poet_idx] = 1
                    else:
                        class_weight[poet_idx] += 1

        max_w = max([class_weight[i] for i in class_weight])
        class_weight = {i: max_w / class_weight[i] for i in class_weight}

        return lines, class_weight

    def build_model(self):
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(self.MAX_LINE_LEN, self.EMB_SIZE)))
        model.add(Bidirectional(LSTM(64, dropout=0.2)))
        model.add(Dense(5, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def init_batch(self, size):
        x = np.zeros(shape=(size, self.MAX_LINE_LEN, self.EMB_SIZE))
        y = np.zeros(shape=(size, 5))

        return x, y

    def get_lines_batch(self, lines):
        x, y = self.init_batch(len(lines))
        for i, tokens in enumerate(lines):
            self.fill_x_row(x, i, tokens)

        return x

    def fill_x_row(self, x, row, tokens):
        ti = 0
        for token in tokens:
            if token.syllables is not None and token.stress_syl is not None and token.tag_vec is not None:
                x[row, ti, :] = [token.syllables, token.stress_syl] + list(token.tag_vec)
                ti += 1

    def get_next_batch(self, lines):
        random.shuffle(lines)
        x, y = self.init_batch(self.BATCH_SIZE)
        row = 0

        for line, tokens, poet in lines:
            self.fill_x_row(x, row, tokens)
            y[row, poet] = 1
            row += 1

            if row == self.BATCH_SIZE:
                yield x, y
                x, y = self.init_batch(self.BATCH_SIZE)
                row = 0

        if row > 0:
            yield x, y

    def train(self, lines, class_weight, epochs = 20):
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            losses = []
            accs = []

            for x, y in self.get_next_batch(lines):
                loss, acc = self.model.train_on_batch(x, y, class_weight=class_weight)
                losses.append(loss)
                accs.append(acc)

            print('Loss: {} Acc: {}\n'.format(np.mean(losses), np.mean(accs)))

    def save(self, path="/data/"):
        self.model.save(path + "poet_line_clf.model")

    def load(self, path="/data/"):
        self.model = keras.models.load_model(path + "poet_line_clf.model")

    def predict_proba(self, tokens_batch, min_proba=0.25):
        x = self.get_lines_batch(tokens_batch)
        y = self.model.predict(x)

        return [{self.POETS[i]: p for i, p in enumerate(row) if p > min_proba} for row in y]


class PoetLineAligner:
    def __init__(self, poet):
        self.poet = poet
        self.model = keras.models.load_model("/data/poet_line_order.model")

    def align_lines(self, lines):
        lines = [[l for l in lines] for lines in itertools.permutations(lines)]
        lines_vec = [[self.poet.poet_lines_vec[l] for l in line] for line in lines]
        predicts = self.model.predict(lines_vec)
        return lines[np.argmax(predicts)]