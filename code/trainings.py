import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
from sklearn.cluster import Birch
from scipy.optimize import linear_sum_assignment


def train_svm_0(path):

    X_train = np.load(path + '/X_train_tf_idf.npy')
    Y_train = np.load(path + '/Y_train_tf_idf.npy')
    X_validation = np.load(path + '/X_validation_tf_idf.npy')
    Y_validation = np.load(path + '/Y_validation_tf_idf.npy')

    print(f'X_train Shape: {X_train.shape}')
    print(f'Y_train Shape: {Y_train.shape}')
    print(f'X_validation Shape: {X_validation.shape}')
    print(f'Y_validation Shape: {Y_validation.shape}')

    svc = SVC(kernel='rbf', C=10.0, random_state=23)
    svc.fit(X_train, Y_train)

    Y_train_pred = svc.predict(X_train)
    Y_validation_pred = svc.predict(X_validation)

    train_accuracy = accuracy_score(Y_train, Y_train_pred)
    validation_accuracy = accuracy_score(Y_validation, Y_validation_pred)

    print(f'SVM Train Accuracy: {train_accuracy}')
    print(f'SVM Validation Accuracy: {validation_accuracy}')


def train_birch_0(path):

    X_train = np.load(path + '/X_train_tf_idf.npy')
    Y_train = np.load(path + '/Y_train_tf_idf.npy')
    X_validation = np.load(path + '/X_validation_tf_idf.npy')
    Y_validation = np.load(path + '/Y_validation_tf_idf.npy')

    label_encoder = joblib.load(path + '/label_encoder_tf_idf.pkl')

    print(f'X_train Shape: {X_train.shape}')
    print(f'Y_train Shape: {Y_train.shape}')
    print(f'X_validation Shape: {X_validation.shape}')
    print(f'Y_validation Shape: {Y_validation.shape}')

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    branching_factors = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_clusters = len(label_encoder.classes_)
    best_model = None
    best_validation_accuracy = -1.0

    for threshold in thresholds:
        for branching_factor in branching_factors:
            birch = Birch(threshold=threshold, branching_factor=branching_factor,
                          n_clusters=num_clusters)
            birch.fit(X_train)

            Y_train_pred = birch.predict(X_train)
            Y_validation_pred = birch.predict(X_validation)

            confusion_matrix_train = confusion_matrix(Y_train, Y_train_pred)
            normalized_confusion_matrix_train = confusion_matrix_train / confusion_matrix_train.sum(axis=1, keepdims=True)

            row_idx_sol, col_idx_sol = linear_sum_assignment(-normalized_confusion_matrix_train)
            from_cluster_id_to_label = {col_idx: row_idx for row_idx, col_idx in zip(row_idx_sol, col_idx_sol)}

            Y_train_pred_labels = np.array([from_cluster_id_to_label[cluster_id] for cluster_id in Y_train_pred])
            Y_validation_pred_labels = np.array([from_cluster_id_to_label[cluster_id] for cluster_id in Y_validation_pred])

            train_accuracy = accuracy_score(Y_train, Y_train_pred_labels)
            validation_accuracy = accuracy_score(Y_validation, Y_validation_pred_labels)

            print(f'BIRCH threshold={threshold} branching_factor={branching_factor}')
            print(f'Train Accuracy: {train_accuracy}')
            print(f'Validation Accuracy: {validation_accuracy}')

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_model = birch

    print(f'Best BIRCH Validation Accuracy: {best_validation_accuracy}')
    joblib.dump(best_model, path + f'/../models/birch_model_tf_idf_{best_validation_accuracy}.pkl')



