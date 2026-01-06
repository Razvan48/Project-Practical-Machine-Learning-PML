import pandas as pd
import matplotlib.pyplot as plt


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
    

    
