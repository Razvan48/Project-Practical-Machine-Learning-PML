import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from skfuzzy.cluster import cmeans
from skfuzzy.cluster import cmeans_predict
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def visualize_birch_0(root_path, model_name, dataset_type):
    # This function does multiple visualizations and interpretations for a trained BIRCH Clustering model on train and validation datasets.
    # root_path is the path to the root folder (see main.py for folder hierarchy).
    # model_name is the name of the saved BIRCH model (see trainings.py for how they are named).
    # dataset_type is {'tf_idf', 'fasttext'}.

    X_train = np.load(root_path + f'/data/X_train_{dataset_type}.npy')
    Y_train = np.load(root_path + f'/data/Y_train_{dataset_type}.npy')
    X_validation = np.load(root_path + f'/data/X_validation_{dataset_type}.npy')
    Y_validation = np.load(root_path + f'/data/Y_validation_{dataset_type}.npy')

    label_encoder = joblib.load(root_path + f'/data/label_encoder_{dataset_type}.pkl')

    print(f'X_train Shape: {X_train.shape}')
    print(f'Y_train Shape: {Y_train.shape}')
    print(f'X_validation Shape: {X_validation.shape}')
    print(f'Y_validation Shape: {Y_validation.shape}')

    num_clusters = len(label_encoder.classes_)
    birch_parameters = joblib.load(root_path + f'/models/{model_name}.pkl')
    birch = Birch(threshold=birch_parameters['threshold'], branching_factor=birch_parameters['branching_factor'], n_clusters=num_clusters)
    birch.fit(X_train)

    Y_train_pred = birch.predict(X_train)
    Y_validation_pred = birch.predict(X_validation)

    confusion_matrix_train = confusion_matrix(Y_train, Y_train_pred)
    normalized_confusion_matrix_train = confusion_matrix_train / confusion_matrix_train.sum(axis=1, keepdims=True)

    # Hungarian Algorithm to map cluster IDs to true labels (used the normalized negated confusion matrix as cost matrix).
    row_idx_sol, col_idx_sol = linear_sum_assignment(-normalized_confusion_matrix_train)
    from_cluster_id_to_label = {col_idx: row_idx for row_idx, col_idx in zip(row_idx_sol, col_idx_sol)}

    Y_train_pred_labels = np.array([from_cluster_id_to_label[cluster_id] for cluster_id in Y_train_pred])
    Y_validation_pred_labels = np.array([from_cluster_id_to_label[cluster_id] for cluster_id in Y_validation_pred])

    # Confusion Matrices for Train and Validation Sets
    confusion_matrix_train_labels = confusion_matrix(Y_train, Y_train_pred_labels)
    confusion_matrix_validation_labels = confusion_matrix(Y_validation, Y_validation_pred_labels)

    confusion_matrix_display_train = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_train_labels,
                                                            display_labels=label_encoder.classes_)
    confusion_matrix_display_validation = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_validation_labels,
                                                                 display_labels=label_encoder.classes_)
    
    # Plot and save confusion matrices
    confusion_matrix_display_train.plot(cmap=plt.cm.Blues)
    plt.title('BIRCH Train Confusion Matrix')
    plt.savefig(root_path + f'/plots/{model_name}_confusion_matrix_train.png')
    plt.show()

    confusion_matrix_display_validation.plot(cmap=plt.cm.Blues)
    plt.title('BIRCH Validation Confusion Matrix')
    plt.savefig(root_path + f'/plots/{model_name}_confusion_matrix_validation.png')
    plt.show()

    # t-SNE Visualizations only for Validation Set (colored with true labels and predicted labels for comparison)
    tsne = TSNE(n_components=2, random_state=23)
    X_validation_tsne = tsne.fit_transform(X_validation)

    plt.figure(figsize=(10, 8))
    for label in range(len(label_encoder.classes_)):
        X_label = X_validation_tsne[Y_validation == label]
        plt.scatter(X_label[:, 0], X_label[:, 1], label=label_encoder.classes_[label], alpha=0.6)
    plt.title('BIRCH Validation t-SNE Visualization (True Labels)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(root_path + f'/plots/{model_name}_tsne_visualization_validation_true.png')
    plt.show()

    plt.figure(figsize=(10, 8))
    for label in range(len(label_encoder.classes_)):
        X_label = X_validation_tsne[Y_validation_pred_labels == label]
        plt.scatter(X_label[:, 0], X_label[:, 1], label=label_encoder.classes_[label], alpha=0.6)
    plt.title('BIRCH Validation t-SNE Visualization (Predicted Labels)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(root_path + f'/plots/{model_name}_tsne_visualization_validation_predicted.png')
    plt.show()

    # Compute and print evaluation metrics
    train_accuracy = accuracy_score(Y_train, Y_train_pred_labels)
    validation_accuracy = accuracy_score(Y_validation, Y_validation_pred_labels)

    train_precision = precision_score(Y_train, Y_train_pred_labels, average=None, zero_division=0)
    validation_precision = precision_score(Y_validation, Y_validation_pred_labels, average=None, zero_division=0)

    train_recall = recall_score(Y_train, Y_train_pred_labels, average=None, zero_division=0)
    validation_recall = recall_score(Y_validation, Y_validation_pred_labels, average=None, zero_division=0)

    train_f1_score = f1_score(Y_train, Y_train_pred_labels, average='weighted', zero_division=0)
    validation_f1_score = f1_score(Y_validation, Y_validation_pred_labels, average='weighted', zero_division=0)

    print(f'Train Accuracy: {train_accuracy}')
    print(f'Validation Accuracy: {validation_accuracy}')

    print(f'Train Precision: {train_precision}')
    print(f'Validation Precision: {validation_precision}')

    print(f'Train Recall: {train_recall}')
    print(f'Validation Recall: {validation_recall}')

    print(f'Train F1-Score: {train_f1_score}')
    print(f'Validation F1-Score: {validation_f1_score}')

    # Top TF-IDF Features per Sentiment (only for TF-IDF Dataset)
    if dataset_type == 'tf_idf':
        tf_idf_vectorizer = joblib.load(root_path + f'/data/tf_idf_vectorizer.pkl')

        TOP_K_MOST_SIGNIFICANT_TF_IDF_FEATURES = 10

        for label in range(len(label_encoder.classes_)):
            # We take all samples predicted to belong to this label.
            X_label = X_validation[Y_validation_pred_labels == label]
            # We compute the mean TF-IDF counts across all these samples.
            tf_idf_counts_mean = np.mean(X_label, axis=0)

            # We take the top K features with highest mean TF-IDF counts.
            top_k_feature_indices = np.argsort(tf_idf_counts_mean)[-TOP_K_MOST_SIGNIFICANT_TF_IDF_FEATURES:][::-1]
            print(f'Top {TOP_K_MOST_SIGNIFICANT_TF_IDF_FEATURES} TF-IDF Features for Emotion {label_encoder.classes_[label]}')
            for feature_index in top_k_feature_indices:
                feature_name = tf_idf_vectorizer.get_feature_names_out()[feature_index]
                feature_value = tf_idf_counts_mean[feature_index]
                print(f'{feature_name} with frequency {feature_value}')


def visualize_fuzzy_c_mean_0(root_path, model_name, dataset_type):
    # This function does multiple visualizations and interpretations for a trained Fuzzy C-Means Clustering model on train and validation datasets.
    # root_path is the path to the root folder (see main.py for folder hierarchy).
    # model_name is the name of the saved Fuzzy C-Means model (see trainings.py for how they are named).
    # dataset_type is {'tf_idf', 'fasttext'}.

    X_train = np.load(root_path + f'/data/X_train_{dataset_type}.npy')
    Y_train = np.load(root_path + f'/data/Y_train_{dataset_type}.npy')
    X_validation = np.load(root_path + f'/data/X_validation_{dataset_type}.npy')
    Y_validation = np.load(root_path + f'/data/Y_validation_{dataset_type}.npy')

    label_encoder = joblib.load(root_path + f'/data/label_encoder_{dataset_type}.pkl')

    print(f'X_train Shape: {X_train.shape}')
    print(f'Y_train Shape: {Y_train.shape}')
    print(f'X_validation Shape: {X_validation.shape}')
    print(f'Y_validation Shape: {Y_validation.shape}')

    num_clusters = len(label_encoder.classes_)
    fuzzy_c_means_parameters = joblib.load(root_path + f'/models/{model_name}.pkl')

    NUM_ITERATIONS = 1024

    cluster_centers, membership_matrix_train, _, _, _, _, _ = cmeans(X_train.T, c=num_clusters,
                                                                     m=fuzzy_c_means_parameters['fuzziness_exponent'],
                                                                     error=fuzzy_c_means_parameters['error'],
                                                                     maxiter=NUM_ITERATIONS, init=None)
    
    Y_train_pred = np.argmax(membership_matrix_train, axis=0)

    membership_matrix_validation, _, _, _, _, _ = cmeans_predict(X_validation.T, cluster_centers,
                                                                 m=fuzzy_c_means_parameters['fuzziness_exponent'],
                                                                 error=fuzzy_c_means_parameters['error'],
                                                                 maxiter=NUM_ITERATIONS)
    
    Y_validation_pred = np.argmax(membership_matrix_validation, axis=0)

    confusion_matrix_train = confusion_matrix(Y_train, Y_train_pred)
    normalized_confusion_matrix_train = confusion_matrix_train / confusion_matrix_train.sum(axis=1, keepdims=True)

    # Hungarian Algorithm to map cluster IDs to true labels (used the normalized negated confusion matrix as cost matrix).
    row_idx_sol, col_idx_sol = linear_sum_assignment(-normalized_confusion_matrix_train)
    from_cluster_id_to_label = {col_idx: row_idx for row_idx, col_idx in zip(row_idx_sol, col_idx_sol)}

    Y_train_pred_labels = np.array([from_cluster_id_to_label[cluster_id] for cluster_id in Y_train_pred])
    Y_validation_pred_labels = np.array([from_cluster_id_to_label[cluster_id] for cluster_id in Y_validation_pred])

    confusion_matrix_train_labels = confusion_matrix(Y_train, Y_train_pred_labels)
    confusion_matrix_validation_labels = confusion_matrix(Y_validation, Y_validation_pred_labels)

    # Plot and save confusion matrices
    confusion_matrix_display_train = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_train_labels,
                                                            display_labels=label_encoder.classes_)
    confusion_matrix_display_validation = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_validation_labels,
                                                                 display_labels=label_encoder.classes_)
    
    confusion_matrix_display_train.plot(cmap=plt.cm.Blues)
    plt.title('Fuzzy C-Means Train Confusion Matrix')
    plt.savefig(root_path + f'/plots/{model_name}_confusion_matrix_train.png')
    plt.show()

    confusion_matrix_display_validation.plot(cmap=plt.cm.Blues)
    plt.title('Fuzzy C-Means Validation Confusion Matrix')
    plt.savefig(root_path + f'/plots/{model_name}_confusion_matrix_validation.png')
    plt.show()

    # t-SNE Visualizations only for Validation Set (colored with true labels and predicted labels for comparison)
    tsne = TSNE(n_components=2, random_state=23)
    X_validation_tsne = tsne.fit_transform(X_validation)

    plt.figure(figsize=(10, 8))
    for label in range(len(label_encoder.classes_)):
        X_label = X_validation_tsne[Y_validation == label]
        plt.scatter(X_label[:, 0], X_label[:, 1], label=label_encoder.classes_[label], alpha=0.6)
    plt.title('Fuzzy C-Means Validation t-SNE Visualization (True Labels)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(root_path + f'/plots/{model_name}_tsne_visualization_validation_true.png')
    plt.show()

    plt.figure(figsize=(10, 8))
    for label in range(len(label_encoder.classes_)):
        X_label = X_validation_tsne[Y_validation_pred_labels == label]
        plt.scatter(X_label[:, 0], X_label[:, 1], label=label_encoder.classes_[label], alpha=0.6)
    plt.title('Fuzzy C-Means Validation t-SNE Visualization (Predicted Labels)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(root_path + f'/plots/{model_name}_tsne_visualization_validation_predicted.png')
    plt.show()

    # Compute and print evaluation metrics
    train_accuracy = accuracy_score(Y_train, Y_train_pred_labels)
    validation_accuracy = accuracy_score(Y_validation, Y_validation_pred_labels)

    train_precision = precision_score(Y_train, Y_train_pred_labels, average=None, zero_division=0)
    validation_precision = precision_score(Y_validation, Y_validation_pred_labels, average=None, zero_division=0)

    train_recall = recall_score(Y_train, Y_train_pred_labels, average=None, zero_division=0)
    validation_recall = recall_score(Y_validation, Y_validation_pred_labels, average=None, zero_division=0)

    train_f1_score = f1_score(Y_train, Y_train_pred_labels, average='weighted', zero_division=0)
    validation_f1_score = f1_score(Y_validation, Y_validation_pred_labels, average='weighted', zero_division=0)

    print(f'Train Accuracy: {train_accuracy}')
    print(f'Validation Accuracy: {validation_accuracy}')

    print(f'Train Precision: {train_precision}')
    print(f'Validation Precision: {validation_precision}')

    print(f'Train Recall: {train_recall}')
    print(f'Validation Recall: {validation_recall}')

    print(f'Train F1-Score: {train_f1_score}')
    print(f'Validation F1-Score: {validation_f1_score}')

    # Top TF-IDF Features per Sentiment (only for TF-IDF Dataset)
    if dataset_type == 'tf_idf':
        tf_idf_vectorizer = joblib.load(root_path + f'/data/tf_idf_vectorizer.pkl')

        TOP_K_MOST_SIGNIFICANT_TF_IDF_FEATURES = 10

        for label in range(len(label_encoder.classes_)):
            # We take all samples predicted to belong to this label.
            X_label = X_validation[Y_validation_pred_labels == label]
            # We compute the mean TF-IDF counts across all these samples.
            tf_idf_counts_mean = np.mean(X_label, axis=0)

            # We take the top K features with highest mean TF-IDF counts.
            top_k_feature_indices = np.argsort(tf_idf_counts_mean)[-TOP_K_MOST_SIGNIFICANT_TF_IDF_FEATURES:][::-1]
            print(f'Top {TOP_K_MOST_SIGNIFICANT_TF_IDF_FEATURES} TF-IDF Features for Emotion {label_encoder.classes_[label]}')
            for feature_index in top_k_feature_indices:
                feature_name = tf_idf_vectorizer.get_feature_names_out()[feature_index]
                feature_value = tf_idf_counts_mean[feature_index]
                print(f'{feature_name} with frequency {feature_value}')
    

