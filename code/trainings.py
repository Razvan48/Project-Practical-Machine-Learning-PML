import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
from sklearn.cluster import Birch


def train_svm_0(path):

    X_train = np.load(path + '/X_train_tf_idf.npy')
    Y_train = np.load(path + '/Y_train_tf_idf.npy')
    X_test = np.load(path + '/X_test_tf_idf.npy')
    Y_test = np.load(path + '/Y_test_tf_idf.npy')

    print(f'X_train Shape: {X_train.shape}')
    print(f'Y_train Shape: {Y_train.shape}')
    print(f'X_test Shape: {X_test.shape}')
    print(f'Y_test Shape: {Y_test.shape}')

    svc = SVC(kernel='rbf', C=10.0, random_state=23)
    svc.fit(X_train, Y_train)

    Y_train_pred = svc.predict(X_train)
    Y_test_pred = svc.predict(X_test)

    train_accuracy = accuracy_score(Y_train, Y_train_pred)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)

    print(f'SVM Train Accuracy: {train_accuracy}')
    print(f'SVM Test Accuracy: {test_accuracy}')


def train_birch_0(path):

    X_train = np.load(path + '/X_train_tf_idf.npy')
    Y_train = np.load(path + '/Y_train_tf_idf.npy')
    X_test = np.load(path + '/X_test_tf_idf.npy')
    Y_test = np.load(path + '/Y_test_tf_idf.npy')

    label_encoder = joblib.load(path + '/label_encoder_tf_idf.pkl')

    print(f'X_train Shape: {X_train.shape}')
    print(f'Y_train Shape: {Y_train.shape}')
    print(f'X_test Shape: {X_test.shape}')
    print(f'Y_test Shape: {Y_test.shape}')

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    branching_factors = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_clusters = len(label_encoder.classes_)

    for threshold in thresholds:
        for branching_factor in branching_factors:
            birch = Birch(threshold=threshold, branching_factor=branching_factor,
                          n_clusters=num_clusters)
            birch.fit(X_train)

            Y_train_pred = birch.predict(X_train)
            Y_test_pred = birch.predict(X_test)

            train_accuracy = accuracy_score(Y_train, Y_train_pred)
            test_accuracy = accuracy_score(Y_test, Y_test_pred)

            print(f'Birch Train Accuracy (threshold={threshold}, branching_factor={branching_factor}): {train_accuracy}')
            print(f'Birch Test Accuracy (threshold={threshold}, branching_factor={branching_factor}): {test_accuracy}')



