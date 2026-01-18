import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import joblib
import fasttext

import dataVisualization


def create_tf_idf_dataset_0(csv_file_path):
    
    csv_file = pd.read_csv(csv_file_path)
    
    emotions = csv_file['Emotion']
    texts = csv_file['Text']

    preprocessed_texts = [dataVisualization.preprocess_text_0(text) for text in texts]

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    tokenized_texts = [dataVisualization.tokenize_text_0(preprocessed_text, stop_words=stop_words) for preprocessed_text in preprocessed_texts]

    final_texts = [' '.join(tokenized_text) for tokenized_text in tokenized_texts]

    TRAIN_PERCENTAGE = 0.7

    X_train, X_validation, Y_train, Y_validation = train_test_split(final_texts, emotions,
                                                        train_size=TRAIN_PERCENTAGE,
                                                        random_state=23, shuffle=True,
                                                        stratify=emotions)
    
    tf_idf_vectorizer = TfidfVectorizer(max_features=128)
    tf_idf_vectorizer.fit(X_train)
    X_train_tf_idf = tf_idf_vectorizer.transform(X_train)
    X_validation_tf_idf = tf_idf_vectorizer.transform(X_validation)

    label_encoder = LabelEncoder()
    label_encoder.fit(Y_train)
    Y_train_encoded = label_encoder.transform(Y_train)
    Y_validation_encoded = label_encoder.transform(Y_validation)

    X_train_tf_idf = X_train_tf_idf.toarray()
    X_validation_tf_idf = X_validation_tf_idf.toarray()

    print(f'X_train_tf_idf Shape: {X_train_tf_idf.shape}')
    print(f'Y_train_encoded Shape: {Y_train_encoded.shape}')
    print(f'X_validation_tf_idf Shape: {X_validation_tf_idf.shape}')
    print(f'Y_validation_encoded Shape: {Y_validation_encoded.shape}')

    np.save('../data/X_train_tf_idf.npy', X_train_tf_idf)
    np.save('../data/Y_train_tf_idf.npy', Y_train_encoded)
    np.save('../data/X_validation_tf_idf.npy', X_validation_tf_idf)
    np.save('../data/Y_validation_tf_idf.npy', Y_validation_encoded)

    os.makedirs('../data', exist_ok=True)
    joblib.dump(label_encoder, '../data/label_encoder_tf_idf.pkl')


def create_fasttext_dataset_0(csv_file_path):
       
    csv_file = pd.read_csv(csv_file_path)
    
    emotions = csv_file['Emotion']
    texts = csv_file['Text']

    preprocessed_texts = [dataVisualization.preprocess_text_0(text) for text in texts]

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    tokenized_texts = [dataVisualization.tokenize_text_0(preprocessed_text, stop_words=stop_words) for preprocessed_text in preprocessed_texts]

    model = fasttext.load_model('../fasttext/cc.en.300.bin')
    model_vocabulary = set(model.get_words())
    model_dimension = model.get_dimension()

    embedded_texts = []
    for idx_text, tokenized_text in enumerate(tokenized_texts):
        embedded_tokens = []
        for token in tokenized_text:
            if token in model_vocabulary:
                embedded_token = model.get_word_vector(token)
                embedded_tokens.append(embedded_token)
        if len(embedded_tokens) > 0:
            embedded_tokens = np.array(embedded_tokens)
            embedded_texts.append(np.mean(embedded_tokens, axis=0))
        else:
            embedded_texts.append(np.zeros(model_dimension))
    embedded_texts = np.array(embedded_texts)

    TRAIN_PERCENTAGE = 0.7

    X_train, X_validation, Y_train, Y_validation = train_test_split(embedded_texts, emotions,
                                                        train_size=TRAIN_PERCENTAGE,
                                                        random_state=23, shuffle=True,
                                                        stratify=emotions)

    label_encoder = LabelEncoder()
    label_encoder.fit(Y_train)
    Y_train_encoded = label_encoder.transform(Y_train)
    Y_validation_encoded = label_encoder.transform(Y_validation)

    print(f'X_train Shape: {X_train.shape}')
    print(f'Y_train_encoded Shape: {Y_train_encoded.shape}')
    print(f'X_validation Shape: {X_validation.shape}')
    print(f'Y_validation_encoded Shape: {Y_validation_encoded.shape}')

    np.save('../data/X_train_fasttext.npy', X_train)
    np.save('../data/Y_train_fasttext.npy', Y_train_encoded)
    np.save('../data/X_validation_fasttext.npy', X_validation)
    np.save('../data/Y_validation_fasttext.npy', Y_validation_encoded)

    os.makedirs('../data', exist_ok=True)
    joblib.dump(label_encoder, '../data/label_encoder_fasttext.pkl')
    




    
