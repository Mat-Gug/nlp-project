import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import spacy
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import re
from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)

nltk.download('stopwords')
stop_words = list(set(STOPWORDS.union(stopwords.words('english'))))
nlp = spacy.load('en_core_web_sm')

def show_barplot(x: str, df: pd.DataFrame):
    
    df1 = df[x].value_counts(normalize=True).mul(100).rename('percent').reset_index()
    
    sns.set(style="whitegrid")
    g = sns.catplot(x=x,y='percent',kind='bar',data=df1,hue=x,palette=["blue","red"])
    g.ax.set_ylim(0,100)

    for p in g.ax.patches:
        txt = str(p.get_height().round(2)) + '%'
        txt_x = (p.get_x() + (p.get_x()+p.get_width()))/2
        txt_y = p.get_height()
        g.ax.annotate(txt, (txt_x,txt_y), ha='center')

    plt.title('Label Distribution', fontsize=18)
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.show()

def _data_cleaner(sentence):
    """
    This function takes a sentence as input and performs the following operations:
    1. Converts the sentence to lowercase.
    2. Removes English stopwords.
    3. Removes the prefix "Subject:" (if present).
    4. Removes punctuation.
    5. Removes digits.
    6. Tokenizes the sentence using spaCy and lemmatizes each token.
    7. Removes extra whitespaces.

    Args:
    - sentence (str): Input sentence to be cleaned.

    Returns:
    - str: Cleaned sentence.
    """
    sentence = sentence.lower()
    sentence = ' '.join(word for word in sentence.split() if word not in stop_words)
    to_remove = "subject:"
    if sentence.startswith(to_remove):
        sentence = sentence.removeprefix(to_remove)
    for c in string.punctuation:
        sentence = sentence.replace(c, " ")
    sentence = re.sub('\d', ' ', sentence)
    document = nlp(sentence)
    sentence = ' '.join(token.lemma_ for token in document)
    sentence = re.sub(' +', ' ', sentence).strip()
    
    return sentence

def clean_data(corpus):
    cleaned_corpus = []
    
    for document in tqdm(corpus, desc="Cleaning data", unit="document"):
        cleaned_corpus.append(_data_cleaner(document))
    
    return cleaned_corpus

def show_wordclouds(x, text_col, df):
    
    labels = df[x].unique()
    
    plt.figure(figsize=(20,10))
    count=1
    for label in labels:
        label_text = ' '.join(df[df['label']==label][text_col])
        wordcloud = WordCloud(width=1500, 
                              height=1200,
                              random_state=0,
                              background_color ='black', 
                              margin=1,
                              stopwords = stop_words,
                              ).generate(label_text)
        plt.subplot(1,2,count)
        plt.axis("off")
        plt.title("Label: " + label,fontsize=18)
        plt.tight_layout(pad=3)
        plt.imshow(wordcloud,interpolation='bilinear')
        count=count+1
    plt.show()

# NON ANCORA USATE!!!!!

def _plot_confusion_matrix(model, data: tuple, title="Train", subplot=121):
    """
    Plot confusion matrix for a classification model.

    Args:
        model: 
            The classification model.
        data (tuple): 
            Tuple containing X and y.
        title (str, optional): 
            Title for the confusion matrix plot. Default is "Train".
        subplot (int, optional): 
            Subplot position for plotting. Default is 121.
    """
    X, y = data
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    # balanced_accuracy = balanced_accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    # f2 = fbeta_score(y, y_pred, beta=2)
    
    df_cm = pd.DataFrame(cm,
                        index=["Negative", "Positive"],
                        columns=["Predicted Negative", "Predicted Positive"]
                        )
    
    plt.subplot(subplot)
    ax = plt.gca()
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14})
    x = 0.5
    ax.text(1, -0.22, title, ha='center', fontsize=18)
    ax.text(x, -0.12, f"Precision={precision:.3f}", ha='center', fontsize=14)
    ax.text(x, -0.03, f"Recall={recall:.3f}", ha='center', fontsize=14)
    ax.text(x + 1, -0.12, f"Accuracy={accuracy:.3f}", ha='center', fontsize=14)
    ax.text(x + 1, -0.03, f"F1 Score={f1:.3f}", ha='center', fontsize=14)

def plot_confusion_matrices(model, data: tuple):
    """
    Plot confusion matrices for a classification model on both training and test data.

    Args:
        model: 
            The classification model.
        data (tuple): 
            Tuple containing X_train, X_test, y_train, and y_test.
    """
    X_train, X_test, y_train, y_test = data
    plt.figure(figsize=(16, 7))
    _plot_confusion_matrix(model, (X_train, y_train), title="Train", subplot=121)
    _plot_confusion_matrix(model, (X_test, y_test), title="Test", subplot=122)
    plt.tight_layout()