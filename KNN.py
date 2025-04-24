import math
from collections import Counter
from Matrix import Matrix

class KNNClassifier:
    def __init__(self, k=1):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X: Matrix, y: list):
        self.X = X
        self.y = y

    def predict(self, X: Matrix) -> list:
        return [self._predict(row) for row in X.data]

    def _predict(self, test_row: list) -> str:
        distances = []
        for i, train_row in enumerate(self.X.data):
            dist = math.dist(test_row, train_row)
            distances.append((dist, self.y[i]))

        k_nearest = sorted(distances)[:self.k]
        labels = [label for (_, label) in k_nearest]
        return Counter(labels).most_common(1)[0][0]

    def score(self, X: Matrix, y: list) -> float:
        predictions = self.predict(X)
        correct = sum(1 for p, t in zip(predictions, y) if p == t)
        return correct / len(y)