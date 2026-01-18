import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import os
import joblib
from sklearn.cluster import Birch
from scipy.optimize import linear_sum_assignment
from skfuzzy.cluster import cmeans
from skfuzzy.cluster import cmeans_predict


def random_0(path, dataset_type):

    X_train = np.load(path + f'/X_train_{dataset_type}.npy')
    Y_train = np.load(path + f'/Y_train_{dataset_type}.npy')
    X_validation = np.load(path + f'/X_validation_{dataset_type}.npy')
    Y_validation = np.load(path + f'/Y_validation_{dataset_type}.npy')

    print(f'X_train Shape: {X_train.shape}')
    print(f'Y_train Shape: {Y_train.shape}')
    print(f'X_validation Shape: {X_validation.shape}')
    print(f'Y_validation Shape: {Y_validation.shape}')

    labels, label_counts = np.unique(Y_train, return_counts=True)
    label_probabilities = label_counts / np.sum(label_counts)

    Y_train_pred = np.random.choice(labels, size=Y_train.shape[0], p=label_probabilities)
    Y_validation_pred = np.random.choice(labels, size=Y_validation.shape[0], p=label_probabilities)

    train_accuracy = accuracy_score(Y_train, Y_train_pred)
    validation_accuracy = accuracy_score(Y_validation, Y_validation_pred)

    print(f'Random Train Accuracy: {train_accuracy}')
    print(f'Random Validation Accuracy: {validation_accuracy}')


def train_svm_0(path, dataset_type):

    X_train = np.load(path + f'/X_train_{dataset_type}.npy')
    Y_train = np.load(path + f'/Y_train_{dataset_type}.npy')
    X_validation = np.load(path + f'/X_validation_{dataset_type}.npy')
    Y_validation = np.load(path + f'/Y_validation_{dataset_type}.npy')

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


def train_random_forest_0(path, dataset_type):

    X_train = np.load(path + f'/X_train_{dataset_type}.npy')
    Y_train = np.load(path + f'/Y_train_{dataset_type}.npy')
    X_validation = np.load(path + f'/X_validation_{dataset_type}.npy')
    Y_validation = np.load(path + f'/Y_validation_{dataset_type}.npy')

    print(f'X_train Shape: {X_train.shape}')
    print(f'Y_train Shape: {Y_train.shape}')
    print(f'X_validation Shape: {X_validation.shape}')
    print(f'Y_validation Shape: {Y_validation.shape}')

    max_depths = [5, 10, 15, 20, 30, 50, 70, 100, 200, None]
    num_estimators = [200, 100, 70, 50, 20, 10]
    best_random_forest = None
    best_validation_accuracy = -1.0

    for max_depth in max_depths:
        for num_estimator in num_estimators:
            random_forest = RandomForestClassifier(n_estimators=num_estimator, max_depth=max_depth, random_state=23)
            random_forest.fit(X_train, Y_train)

            Y_train_pred = random_forest.predict(X_train)
            Y_validation_pred = random_forest.predict(X_validation)

            train_accuracy = accuracy_score(Y_train, Y_train_pred)
            validation_accuracy = accuracy_score(Y_validation, Y_validation_pred)

            print(f'Random Forest max_depth={max_depth} num_estimator={num_estimator}')
            print(f'Train Accuracy: {train_accuracy}')
            print(f'Validation Accuracy: {validation_accuracy}')

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_random_forest = random_forest

    print(f'Best Random Forest Validation Accuracy: {best_validation_accuracy}')
    os.makedirs(path + '/../models', exist_ok=True)
    joblib.dump(best_random_forest, path + f'/../models/random_forest_{dataset_type}_{best_validation_accuracy}.pkl')
    

def train_birch_0(path, dataset_type):

    X_train = np.load(path + f'/X_train_{dataset_type}.npy')
    Y_train = np.load(path + f'/Y_train_{dataset_type}.npy')
    X_validation = np.load(path + f'/X_validation_{dataset_type}.npy')
    Y_validation = np.load(path + f'/Y_validation_{dataset_type}.npy')

    label_encoder = joblib.load(path + f'/label_encoder_{dataset_type}.pkl')

    print(f'X_train Shape: {X_train.shape}')
    print(f'Y_train Shape: {Y_train.shape}')
    print(f'X_validation Shape: {X_validation.shape}')
    print(f'Y_validation Shape: {Y_validation.shape}')

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    branching_factors = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_clusters = len(label_encoder.classes_)
    best_birch = None
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
                best_birch = birch

    print(f'Best BIRCH Validation Accuracy: {best_validation_accuracy}')
    os.makedirs(path + '/../models', exist_ok=True)
    joblib.dump(best_birch, path + f'/../models/birch_{dataset_type}_{best_validation_accuracy}.pkl')


def train_fuzzy_c_mean_0(path, dataset_type):
    
    X_train = np.load(path + f'/X_train_{dataset_type}.npy')
    Y_train = np.load(path + f'/Y_train_{dataset_type}.npy')
    X_validation = np.load(path + f'/X_validation_{dataset_type}.npy')
    Y_validation = np.load(path + f'/Y_validation_{dataset_type}.npy')

    label_encoder = joblib.load(path + f'/label_encoder_{dataset_type}.pkl')

    print(f'X_train Shape: {X_train.shape}')
    print(f'Y_train Shape: {Y_train.shape}')
    print(f'X_validation Shape: {X_validation.shape}')
    print(f'Y_validation Shape: {Y_validation.shape}')

    num_clusters = len(label_encoder.classes_)
    fuzziness_exponents = [3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5]
    errors = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    max_iterations = [1000, 750, 500, 250, 100, 75, 50, 25, 10]
    best_fuzzy_c_means = None
    best_validation_accuracy = -1.0

    for fuzziness_exponent in fuzziness_exponents:
        for error in errors:
            for max_iteration in max_iterations:
                cluster_centers, membership_matrix_train, _, _, _, _, _ = cmeans(X_train.T, c=num_clusters, m=fuzziness_exponent, error=error, maxiter=max_iteration, init=None)
                
                Y_train_pred = np.argmax(membership_matrix_train, axis=0)

                membership_matrix_validation, _, _, _, _, _ = cmeans_predict(X_validation.T, cluster_centers, m=fuzziness_exponent, error=error, maxiter=max_iteration)

                Y_validation_pred = np.argmax(membership_matrix_validation, axis=0)

                confusion_matrix_train = confusion_matrix(Y_train, Y_train_pred)
                normalized_confusion_matrix_train = confusion_matrix_train / confusion_matrix_train.sum(axis=1, keepdims=True)

                row_idx_sol, col_idx_sol = linear_sum_assignment(-normalized_confusion_matrix_train)
                from_cluster_id_to_label = {col_idx: row_idx for row_idx, col_idx in zip(row_idx_sol, col_idx_sol)}

                Y_train_pred_labels = np.array([from_cluster_id_to_label[cluster_id] for cluster_id in Y_train_pred])
                Y_validation_pred_labels = np.array([from_cluster_id_to_label[cluster_id] for cluster_id in Y_validation_pred])

                train_accuracy = accuracy_score(Y_train, Y_train_pred_labels)
                validation_accuracy = accuracy_score(Y_validation, Y_validation_pred_labels)

                print(f'Fuzzy C-Means fuzziness_exponent={fuzziness_exponent} error={error} max_iteration={max_iteration}')
                print(f'Train Accuracy: {train_accuracy}')
                print(f'Validation Accuracy: {validation_accuracy}')

                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_fuzzy_c_means = (cluster_centers, fuzziness_exponent, error, max_iteration)

    print(f'Best Fuzzy C-Means Validation Accuracy: {best_validation_accuracy}')
    os.makedirs(path + '/../models', exist_ok=True)
    joblib.dump(best_fuzzy_c_means, path + f'/../models/fuzzy_c_means_{dataset_type}_{best_validation_accuracy}.pkl')







