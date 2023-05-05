# Bringing packages onto the path
import sys
sys.path.append('../..')


from simple_ml.simple_nn import MLPClassifier
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split


def test_mlp_classifier():
    print('MLPClassifier on binary classification dataset')
    print('')
    X, y = make_classification(n_samples=2000, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
    clf = MLPClassifier(epochs=300)
    clf.fit(X_train, y_train)
    print('')
    print(f'Actual: {y_test[:10]}')
    print(f'Predicted: {clf.predict(X_test[:10, :])}')
    print(f'Test accuracy: {clf.score(X_test, y_test)}')
    print('')
    print('-------------------------------------------------')
    print('MLPClassifier on multiclass classification dataset (MNIST)')
    print('')
    mnist = load_digits()
    X, y = mnist.data, mnist.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
    clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100), epochs=300)
    clf.fit(X_train, y_train)
    print('')
    print(f'Actual: {y_test[:10]}')
    print(f'Predicted: {clf.predict(X_test[:10, :])}')
    print(f'Test accuracy: {clf.score(X_test, y_test)}')



if "__main__" == __name__:
    print('')
    print("Testing MLPClassifier...")
    print('')
    test_mlp_classifier()
    print('')