import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.phrases import Phrases

def lda_preprocessing(emails):

    cleaned_tokens = [[word for word in email.split() if len(word) > 1] for email in emails]
    bigram = Phrases(cleaned_tokens, min_count=20)
    trigram = Phrases(bigram[cleaned_tokens], min_count=20)
    cleaned_tokens_with_bigrams_trigrams = [trigram[bigram[token]] for token in cleaned_tokens]
    dictionary = Dictionary(cleaned_tokens_with_bigrams_trigrams)
    print("Dictionary size before filtering:", len(dictionary))
    dictionary.filter_extremes(no_below=5, no_above=0.95)
    print("Dictionary size before filtering:", len(dictionary))
    corpus = [dictionary.doc2bow(text) for text in cleaned_tokens_with_bigrams_trigrams]
    
    return cleaned_tokens_with_bigrams_trigrams, dictionary, corpus

def plot_coherence_values(texts, corpus, dictionary, topics):
    
    coherences = []

    for num_topics in tqdm(topics, 
                           desc="Training and evaluating LDA models",
                           unit="num_topics"):
        lda = LdaMulticore(corpus=corpus, 
                           id2word=dictionary, 
                           num_topics=num_topics, 
                           passes=10)
        coherence_model_lda = CoherenceModel(model=lda,
                                             texts=texts,
                                             dictionary=dictionary,
                                             coherence='c_v')
        coherence = coherence_model_lda.get_coherence()
        coherences.append(coherence)

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=topics, y=coherences, marker='o')
    plt.xticks(topics)

    plt.title('Coherence Score vs. Number of Topics', fontsize=18)
    plt.xlabel('Number of Topics', fontsize=14)
    plt.ylabel('Coherence Score', fontsize=14)

    plt.show()