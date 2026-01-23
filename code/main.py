import dataVisualization
import datasetCreation
import trainings
import visualizations

import warnings
warnings.filterwarnings('ignore')


# dataVisualization.is_data_valid('../data/emotion_dataset_raw.csv')
# dataVisualization.visualize_data_0('../data/emotion_dataset_raw.csv')
dataVisualization.visualize_data_1('../data/emotion_dataset_raw.csv')

# datasetCreation.create_tf_idf_dataset_0('../data/emotion_dataset_raw.csv')
# datasetCreation.create_fasttext_dataset_0('../data/emotion_dataset_raw.csv')



# trainings.random_0('../data', dataset_type='tf_idf')

# trainings.train_random_forest_0('../data', dataset_type='tf_idf')

# trainings.train_birch_0('../data', dataset_type='tf_idf')
# trainings.train_fuzzy_c_mean_0('../data', dataset_type='tf_idf')



# trainings.random_0('../data', dataset_type='fasttext')

# trainings.train_random_forest_0('../data', dataset_type='fasttext')

# trainings.train_birch_0('../data', dataset_type='fasttext')
# trainings.train_fuzzy_c_mean_0('../data', dataset_type='fasttext')


# visualizations.visualize_birch_0('..', model_name='birch_tf_idf_0.21613335888101168', dataset_type='tf_idf')
# visualizations.visualize_fuzzy_c_mean_0('..', model_name='fuzzy_c_means_fasttext_0.18106917033914544', dataset_type='fasttext')






