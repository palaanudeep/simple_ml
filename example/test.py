# Bringing packages onto the path
import sys
sys.path.append('..')


from simple_nn import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def test_mlp_classifier():
    X, y = make_classification(n_samples=100, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
    y_train = y_train.reshape(-1, 1)
    clf = MLPClassifier(epochs=300)
    clf.fit(X_train, y_train)
    print(f'Actual: {y_test[:5]}')
    print(f'Predicted: {clf.predict(X_test[:5, :])}')
    print(f'Total test score: {clf.score(X_test, y_test)}')


if "__main__" == __name__:
    print("Testing MLPClassifier...")
    test_mlp_classifier()
