import dataVisualization
import datasetCreation
import trainings
import visualizations

import warnings
warnings.filterwarnings('ignore')


# dataVisualization.is_data_valid('../data/emotion_dataset_raw.csv')
# dataVisualization.visualize_data_0('../data/emotion_dataset_raw.csv')
# dataVisualization.visualize_data_1('../data/emotion_dataset_raw.csv')

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


# visualizations.visualize_birch_0('..', model_name='birch_tf_idf_0.2263843648208469', dataset_type='tf_idf')
# visualizations.visualize_fuzzy_c_mean_0('..', model_name='fuzzy_c_means_tf_idf_0.24822762981414065', dataset_type='tf_idf')

# visualizations.visualize_birch_0('..', model_name='birch_fasttext_0.10011496455259629', dataset_type='fasttext')
# visualizations.visualize_fuzzy_c_mean_0('..', model_name='fuzzy_c_means_fasttext_0.18729641693811075', dataset_type='fasttext')










