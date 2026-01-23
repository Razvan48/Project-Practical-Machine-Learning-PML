import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords


def visualize_data_0(csv_file_path):
    
    csv_file = pd.read_csv(csv_file_path)
    
    emotions = csv_file['Emotion']
    texts = csv_file['Text']
    
    num_texts_per_emotion = dict()
    for emotion, text in zip(emotions, texts):
        if emotion not in num_texts_per_emotion:
            num_texts_per_emotion[emotion] = 0
        num_texts_per_emotion[emotion] += 1
        
    print(f'Num Texts per Emotion: {num_texts_per_emotion}')
    
    plt.bar(num_texts_per_emotion.keys(), num_texts_per_emotion.values())
    plt.title('Number of Texts per Emotion')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Texts')
    plt.savefig('../plots/num_texts_per_emotion.png')
    plt.show()
    
    plt.pie(num_texts_per_emotion.values(), labels=num_texts_per_emotion.keys(), autopct='%1.2f%%')
    plt.title('Percentages of Texts per Emotion')
    plt.savefig('../plots/percentages_texts_per_emotion.png')
    plt.show()
    
    
def preprocess_text_0(text):
    preprocessed_text = text.strip().lower()
    # preprocessed_text = re.sub(r'[^a-z.,?!\s]', '', preprocessed_text)
    preprocessed_text = re.sub(r'[^a-z\s]', '', preprocessed_text)
    return preprocessed_text


def tokenize_text_0(preprocessed_text, stop_words=set()):
    # tokenized_text = re.findall(r'[a-z]+|[.,?!]', preprocessed_text)
    tokenized_text = re.findall(r'[a-z]+', preprocessed_text)
    tokenized_text = [token for token in tokenized_text if token not in stop_words]
    return tokenized_text


def visualize_data_1(csv_file_path):
    
    csv_file = pd.read_csv(csv_file_path)
    
    emotions = csv_file['Emotion']
    texts = csv_file['Text']
    
    preprocessed_texts = [preprocess_text_0(text) for text in texts]
    
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    tokenized_texts = [tokenize_text_0(preprocessed_text, stop_words=stop_words) for preprocessed_text in preprocessed_texts]
    
    num_tokens_per_emotion = dict()
    num_texts_per_emotion = dict()
    tokens_per_emotion = dict()
    for emotion, tokenized_text in zip(emotions, tokenized_texts):
        if emotion not in num_tokens_per_emotion:
            num_tokens_per_emotion[emotion] = 0
            num_texts_per_emotion[emotion] = 0
            tokens_per_emotion[emotion] = []
        num_tokens_per_emotion[emotion] += len(tokenized_text)
        num_texts_per_emotion[emotion] += 1
        tokens_per_emotion[emotion].extend(tokenized_text)
        
    avg_num_tokens_per_emotion = dict()
    for emotion in num_tokens_per_emotion.keys():
        avg_num_tokens_per_emotion[emotion] = num_tokens_per_emotion[emotion] / num_texts_per_emotion[emotion]
        
    TOP_K_MOST_USED_TOKENS = 10
    top_k_most_used_tokens_per_emotion = dict()
    for emotion in tokens_per_emotion.keys():
        token_freq = dict()
        for token in tokens_per_emotion[emotion]:
            if token not in token_freq:
                token_freq[token] = 0
            token_freq[token] += 1
        sorted_token_freq = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        top_k_most_used_tokens_per_emotion[emotion] = sorted_token_freq[:TOP_K_MOST_USED_TOKENS]
        
    print(f'Top {TOP_K_MOST_USED_TOKENS} Most Used Tokens per Emotion')
    for emotion, top_k_tokens in top_k_most_used_tokens_per_emotion.items():
        print(f'Emotion: {emotion}')
        for token, freq in top_k_tokens:
            print(f'Token: {token} Frequency: {freq}')
    
    plt.bar(avg_num_tokens_per_emotion.keys(), avg_num_tokens_per_emotion.values())
    plt.title('Average Number of Tokens per Emotion')
    plt.xlabel('Emotion')
    plt.ylabel('Average Number of Tokens')
    plt.savefig('../plots/avg_num_tokens_per_emotion.png')
    plt.show()


def is_data_valid(csv_file_path):

    csv_file = pd.read_csv(csv_file_path)

    num_samples = len(csv_file)
    print(f'Number of Samples in Dataset: {num_samples}')
    
    emotions = csv_file['Emotion']
    texts = csv_file['Text']

    for text in texts:
        if len(text.strip()) == 0:
            print('Invalid Data: Text of Length 0 Found')
            return False
        
    for emotion in emotions:
        if emotion not in ['neutral', 'joy', 'sadness', 'fear', 'surprise', 'anger', 'shame', 'disgust']:
            print(f'Invalid Data: Unknown Emotion {emotion} Found')
            return False
        
    print('Data is Valid')
    return True





    

    
