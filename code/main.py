import dataVisualization
import datasetCreation
import trainings
import visualizations

import warnings
warnings.filterwarnings('ignore')


# The project folder hierarchy is the following:
# root -> code -> main.py/trainings.py/datasetCreation.py/dataVisualization.py/visualizations.py
#   |
#   |---> data -> emotion_dataset_raw.csv (original dataset) (here in the data folder we also save created datasets, TF-IDF and Fasttext)
#   |
#   |---> fasttext -> cc.en.300.bin (pretrained Fasttext model)
#   |
#   |---> models -> here we save trained models
#   |
#   |---> plots -> here we save generated plots


# dataVisualization.is_data_valid('../data/emotion_dataset_raw.csv') # This checks if the data is valid.

# Visualizations of the raw data (just need .csv file)
# dataVisualization.visualize_data_0('../data/emotion_dataset_raw.csv')
# dataVisualization.visualize_data_1('../data/emotion_dataset_raw.csv')

# Creation of Datasets (TF-IDF and Fasttext)
# datasetCreation.create_tf_idf_dataset_0('../data/emotion_dataset_raw.csv') # Creates TF-IDF Dataset
# datasetCreation.create_fasttext_dataset_0('../data/emotion_dataset_raw.csv') # Creates Fasttext Dataset


# Trainings of various models on the created datasets (make sure datasets are created first)

# trainings.random_0('../data', dataset_type='tf_idf') # Baseline Random Classifier for TF-IDF Dataset

# trainings.train_random_forest_0('../data', dataset_type='tf_idf') # Random Forest Classifier for TF-IDF Dataset

# trainings.train_birch_0('../data', dataset_type='tf_idf') # Birch Clustering for TF-IDF Dataset
# trainings.train_fuzzy_c_mean_0('../data', dataset_type='tf_idf') # Fuzzy C-Means Clustering for TF-IDF Dataset



# trainings.random_0('../data', dataset_type='fasttext') # Baseline Random Classifier for Fasttext Dataset

# trainings.train_random_forest_0('../data', dataset_type='fasttext') # Random Forest Classifier for Fasttext Dataset

# trainings.train_birch_0('../data', dataset_type='fasttext') # Birch Clustering for Fasttext Dataset
# trainings.train_fuzzy_c_mean_0('../data', dataset_type='fasttext') # Fuzzy C-Means Clustering for Fasttext Dataset


# Visualizations of the trained models and their results (make sure models are trained first and datasets are created)
# visualizations.visualize_birch_0('..', model_name='birch_tf_idf_0.2263843648208469', dataset_type='tf_idf') # Birch Clustering for TF-IDF Dataset
# visualizations.visualize_fuzzy_c_mean_0('..', model_name='fuzzy_c_means_tf_idf_0.24822762981414065', dataset_type='tf_idf') # Fuzzy C-Means Clustering for TF-IDF Dataset

# visualizations.visualize_birch_0('..', model_name='birch_fasttext_0.10011496455259629', dataset_type='fasttext') # Birch Clustering for Fasttext Dataset
# visualizations.visualize_fuzzy_c_mean_0('..', model_name='fuzzy_c_means_fasttext_0.18729641693811075', dataset_type='fasttext') # Fuzzy C-Means Clustering for Fasttext Dataset










