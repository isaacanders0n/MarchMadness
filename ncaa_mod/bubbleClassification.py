import pandas as pd
import cleaning as c




PATH = 'data'

class BubbleClassifier():
    def __init__(self, path):
        self.path = path
        self.df = c.read_and_clean(self.path)
        self.df = self.df.dropna()
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
        self.model = None

    def train(self, model):
        self.model = model
        self.model.fit(self.X, self.y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
    


def main():




    return