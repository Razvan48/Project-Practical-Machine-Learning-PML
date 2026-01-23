import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import os
import joblib
from sklearn.cluster import Birch
from scipy.optimize import linear_sum_assignment
from skfuzzy.cluster import cmeans
from skfuzzy.cluster import cmeans_predict


RECURSION_LIMIT = 50000

import sys
sys.setrecursionlimit(RECURSION_LIMIT)


def random_0(path, dataset_type):

    X_train = np.load(path + f'/X_train_{dataset_type}.npy')
    Y_train = np.load(path + f'/Y_train_{dataset_type}.npy')
    X_validation = np.load(path + f'/X_validation_{dataset_type}.npy')
    Y_validation = np.load(path + f'/Y_validation_{dataset_type}.npy')

    print(f'X_train Shape: {X_train.shape}')
    print(f'Y_train Shape: {Y_train.shape}')
    print(f'X_validation Shape: {X_validation.shape}')
    print(f'Y_validation Shape: {Y_validation.shape}')

    labels = np.unique(Y_train)

    Y_train_pred = np.random.choice(labels, size=Y_train.shape[0])
    Y_validation_pred = np.random.choice(labels, size=Y_validation.shape[0])

    train_accuracy = accuracy_score(Y_train, Y_train_pred)
    validation_accuracy = accuracy_score(Y_validation, Y_validation_pred)

    print(f'Random Train Accuracy: {train_accuracy}')
    print(f'Random Validation Accuracy: {validation_accuracy}')


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

    # permutation = np.random.permutation(2500)
    # X_train = X_train[permutation]
    # Y_train = Y_train[permutation]

    label_encoder = joblib.load(path + f'/label_encoder_{dataset_type}.pkl')

    print(f'X_train Shape: {X_train.shape}')
    print(f'Y_train Shape: {Y_train.shape}')
    print(f'X_validation Shape: {X_validation.shape}')
    print(f'Y_validation Shape: {Y_validation.shape}')

    # thresholds = [0.4, 0.3, 0.2, 0.1, 0.05]
    # branching_factors = [20, 40, 60, 80, 100]
    
    # thresholds = [0.5, 0.52, 0.55, 0.57, 0.6, 0.62, 0.65, 0.67, 0.7]
    # branching_factors = [70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90]

    thresholds = [0.2]
    branching_factors = [20]

    num_clusters = len(label_encoder.classes_)
    best_birch = None
    best_threshold = None
    best_branching_factor = None
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

            train_nmi = normalized_mutual_info_score(Y_train, Y_train_pred_labels)
            validation_nmi = normalized_mutual_info_score(Y_validation, Y_validation_pred_labels)

            if len(np.unique(Y_train_pred)) >= 2:
                train_silhouette_score = silhouette_score(X_train, Y_train_pred)
            else:
                train_silhouette_score = None
            if len(np.unique(Y_validation_pred)) >= 2:
                validation_silhouette_score = silhouette_score(X_validation, Y_validation_pred)
            else:
                validation_silhouette_score = None

            print(f'BIRCH threshold={threshold} branching_factor={branching_factor}')
            print(f'Train Accuracy: {train_accuracy}')
            print(f'Validation Accuracy: {validation_accuracy}')
            print(f'Train NMI: {train_nmi}')
            print(f'Validation NMI: {validation_nmi}')
            print(f'Train Silhouette Score: {train_silhouette_score}')
            print(f'Validation Silhouette Score: {validation_silhouette_score}')

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_birch = birch
                best_threshold = threshold
                best_branching_factor = branching_factor

    print(f'Best BIRCH Validation Accuracy: {best_validation_accuracy}')
    os.makedirs(path + '/../models', exist_ok=True)
    best_birch_parameters = {
        'threshold': best_threshold,
        'branching_factor': best_branching_factor
    }
    joblib.dump(best_birch_parameters, path + f'/../models/birch_{dataset_type}_{best_validation_accuracy}.pkl')


def train_fuzzy_c_mean_0(path, dataset_type):
    
    X_train = np.load(path + f'/X_train_{dataset_type}.npy')
    Y_train = np.load(path + f'/Y_train_{dataset_type}.npy')
    X_validation = np.load(path + f'/X_validation_{dataset_type}.npy')
    Y_validation = np.load(path + f'/Y_validation_{dataset_type}.npy')

    # permutation = np.random.permutation(2500)
    # X_train = X_train[permutation]
    # Y_train = Y_train[permutation]

    label_encoder = joblib.load(path + f'/label_encoder_{dataset_type}.pkl')

    print(f'X_train Shape: {X_train.shape}')
    print(f'Y_train Shape: {Y_train.shape}')
    print(f'X_validation Shape: {X_validation.shape}')
    print(f'Y_validation Shape: {Y_validation.shape}')

    num_clusters = len(label_encoder.classes_)
    # fuzziness_exponents = [1.25, 1.2, 1.15, 1.1, 1.05, 1.03]
    # errors = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    # fuzziness_exponents = [1.03, 1.02, 1.01, 1.005]
    # errors = [1e-3, 1e-4, 1e-5, 1e-6]

    fuzziness_exponents = [1.05]
    errors = [1e-4]

    best_fuzzy_c_means = None
    best_validation_accuracy = -1.0

    NUM_ITERATIONS = 1024

    for fuzziness_exponent in fuzziness_exponents:
        for error in errors:
            cluster_centers, membership_matrix_train, _, _, _, _, _ = cmeans(X_train.T, c=num_clusters, m=fuzziness_exponent, error=error, maxiter=NUM_ITERATIONS, init=None)
            
            Y_train_pred = np.argmax(membership_matrix_train, axis=0)

            membership_matrix_validation, _, _, _, _, _ = cmeans_predict(X_validation.T, cluster_centers, m=fuzziness_exponent, error=error, maxiter=NUM_ITERATIONS)

            Y_validation_pred = np.argmax(membership_matrix_validation, axis=0)

            confusion_matrix_train = confusion_matrix(Y_train, Y_train_pred)
            normalized_confusion_matrix_train = confusion_matrix_train / confusion_matrix_train.sum(axis=1, keepdims=True)

            row_idx_sol, col_idx_sol = linear_sum_assignment(-normalized_confusion_matrix_train)
            from_cluster_id_to_label = {col_idx: row_idx for row_idx, col_idx in zip(row_idx_sol, col_idx_sol)}

            Y_train_pred_labels = np.array([from_cluster_id_to_label[cluster_id] for cluster_id in Y_train_pred])
            Y_validation_pred_labels = np.array([from_cluster_id_to_label[cluster_id] for cluster_id in Y_validation_pred])

            train_accuracy = accuracy_score(Y_train, Y_train_pred_labels)
            validation_accuracy = accuracy_score(Y_validation, Y_validation_pred_labels)

            train_nmi = normalized_mutual_info_score(Y_train, Y_train_pred_labels)
            validation_nmi = normalized_mutual_info_score(Y_validation, Y_validation_pred_labels)

            if len(np.unique(Y_train_pred)) >= 2:
                train_silhouette_score = silhouette_score(X_train, Y_train_pred)
            else:
                train_silhouette_score = None
            if len(np.unique(Y_validation_pred)) >= 2:
                validation_silhouette_score = silhouette_score(X_validation, Y_validation_pred)
            else:
                validation_silhouette_score = None

            print(f'Fuzzy C-Means fuzziness_exponent={fuzziness_exponent} error={error}')
            print(f'Train Accuracy: {train_accuracy}')
            print(f'Validation Accuracy: {validation_accuracy}')
            print(f'Train NMI: {train_nmi}')
            print(f'Validation NMI: {validation_nmi}')
            print(f'Train Silhouette Score: {train_silhouette_score}')
            print(f'Validation Silhouette Score: {validation_silhouette_score}')

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_fuzzy_c_means = (cluster_centers, fuzziness_exponent, error)

    print(f'Best Fuzzy C-Means Validation Accuracy: {best_validation_accuracy}')
    os.makedirs(path + '/../models', exist_ok=True)
    best_fuzzy_c_means_parameters = {
        'fuzziness_exponent': best_fuzzy_c_means[1],
        'error': best_fuzzy_c_means[2]
    }
    joblib.dump(best_fuzzy_c_means_parameters, path + f'/../models/fuzzy_c_means_{dataset_type}_{best_validation_accuracy}.pkl')







