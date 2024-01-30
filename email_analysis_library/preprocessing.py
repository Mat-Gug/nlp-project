import re
import string
import spacy
import nltk
from nltk.corpus import stopwords
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS

nltk.download('stopwords')
stop_words = list(set(STOPWORDS.union(stopwords.words('english'))))
nlp = spacy.load('en_core_web_trf', disable=['parser'])

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