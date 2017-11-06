#!/usr/bin/env python3

import sys
import time
import re
import string

import matplotlib
# Use a backend that doesn't require X server.
matplotlib.use('Agg')
from matplotlib import pyplot

from bs4 import BeautifulSoup
import requests

import numpy as np

import networkx as nx

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE

INTRODUCTION_SUMMARY_LENGTH = 2
PAGE_SUMMARY_LENGTH = 4
TOPIC_SUMMARY_LENGTH = 1


def download(url):
    while True:
        try:
            r = requests.get(url)
        except requests.exceptions.ConnectionError as e:
            # retry once.
            print(e)
            print('Retrying...')
            time.sleep(1)
        else:
            break

    return r.text


def load_from_disk(file):
    with open(file, 'r') as f:
        return f.read()


def validate_paragraph(text):
    """
    Parsing HTML is notoriously tricky. This function does some very elementary
    tests to check if the text given really could be a paragraph.
    """
    if '{' in text:
        return False

    if len(text) < 10:
        return False

    return True


def tokenize_sentences(text):
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(text)
    return list(set(sentences))


def tokenize_tricky_fullstops(text):
    """
    Some abbreviations seem to confuse the tokenizer.
    """
    text = re.sub('\[[^\]]*\]', '', text)
    text = re.sub('(č\.)\s+', 'č.', text)
    #text = re.sub('(s\.)\s+', 's.', text)
    text = re.sub('(p\.)\s+', 'p.', text)
    text = re.sub('(mj\.)\s+', 'mj.', text)
    text = re.sub('(tzv\.)\s+', 'tzv.', text)
    text = re.sub('(např\.)\s+', 'např.', text)
    text = re.sub('(\s+\d+)\s+', '\1', text)
    return text


def prepare_paragraph(text):
    """
    Sanitize text. Remove punctuation, newlines, etc. Replace words with their
    stems.
    """
    translator = str.maketrans({key: None for key in string.punctuation})
    text = text.translate(translator)

    text = text.replace('\n', ' ')
    text = text.replace('  ', ' ')
    words = text.split()

    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(x) for x in words]

    no_numbers = []
    for w in stems:
        if w.isalpha():
            no_numbers.append(w)

    return ' '.join(no_numbers)


def summarize(text, X=None, max_length=10):
    """
    Do the summarization itself, using the PageRank algorithm.

    Accepts either text itself, or a list of sentences with corresponding
    LSA features.

    Returns the most characteristic sentences.
    """

    if not X:
        sentences = tokenize_sentences(text)
        X = get_features_lsa(sentences)
    else:
        sentences = text

    if len(sentences) > 2000:
        # We don't have time for that...
        return ''

    similarity_graph = linear_kernel(X)

    nx_graph = nx.from_numpy_matrix(similarity_graph)
    scores = nx.pagerank_numpy(nx_graph)
    most_characteristic = sorted(((scores[i], s) for i, s in enumerate(sentences)),
                                 reverse=True)
    most_characteristic = [x[1] for x in most_characteristic]

    # Remove sentences that start with "this" and similar.
    cleaned = []
    for sentence in most_characteristic:
        skip_sentence = False
        words = sentence.split()
        for w in words[:3]:
            if w.lower() in ['this', 'that', 'thes']:
                skip_sentence = True
        if not skip_sentence:
            cleaned.append(sentence)

    return cleaned[:max_length]


def find_paragraphs(html):
    soup = BeautifulSoup(html, 'html5lib')
    paragraphs = soup.find_all('p')

    cleaned = []
    for p in paragraphs:
        text = p.get_text().strip()
        if validate_paragraph(text):
            text = tokenize_tricky_fullstops(text)
            cleaned.append(text)

    if not cleaned:
        # No <p> tags. Let's try \n\n.
        text = soup.get_text()
        paragraphs = text.split('\n\n')
        for p in paragraphs:
            text = p.strip()
            if validate_paragraph(text):
                text = tokenize_tricky_fullstops(text)
                cleaned.append(text)

    return cleaned


def get_stopwords():
    stop_words = set(stopwords.words('english'))

    additions = {
                 'get',
                 'know',
                 'may',
                 'bit',
                 'ly',
                 'www',
                 'http',
                 'https',
                 'gl',
                 'goo',
                 'com',
                 'en',
                 'el',
                 'la',
                 'ofa',
                 'bo',
                 'cz',
                 'na',
                 'se',
                 'za',
                 'si',
                 'pro',
                 'je',
                 'us',
                 'ow',
                 'ke',
                 'jak',
                }
    stop_words |= additions
    return stop_words


def get_features_lsa(sentences):
    """
    The key feature of this tool. Preprocess the sentences using
    the Latent Semantic Analysis.
    """
    features = tfidf(sentences)
    min_dimensions = min(features.shape)
    if min_dimensions > 400:
        svd_components = 80
    elif min_dimensions > 80:
        svd_components = int(min_dimensions / 2)
    else:
        svd_components = 0

    normalizer = Normalizer(copy=False)
    if svd_components > 0:
        svd = TruncatedSVD(svd_components, algorithm='arpack', tol=0)
        lsa = make_pipeline(svd, normalizer)
    else:
        lsa = normalizer
    X = lsa.fit_transform(features)

    if np.ndarray != type(X):
        X = X.toarray()
    return X


def get_predictor(X, n_clusters):
    """
    The clustering predictor is used in different places, hence it's defined
    only once.
    """
    knn_graph = kneighbors_graph(X, 5, include_self=False, n_jobs=-1)
    return AgglomerativeClustering(n_clusters=n_clusters,
                                   connectivity=knn_graph, linkage='ward',
                                   affinity='euclidean')


def best_number_clusters(X):
    """
    Figure out which number of clusters gives the best silhouette_score.
    """
    scores = []

    print(time.ctime(), 'best_number_clusters. X.shape:', X.shape)
    cluster_counts = range(2, 5, 1)
    if max(X.shape) > 1000:
        cluster_counts = range(3, 9, 1)

    try:
        for n in cluster_counts:
            predictor = get_predictor(X, n)
            labels = predictor.fit_predict(X)

            scores.append(silhouette_score(X, labels))
    except KeyboardInterrupt:
        pass

    index = scores.index(max(scores))
    return cluster_counts[index]


def organize_clusters(paragraphs, labels, X):
    """
    Order the clusters by average position of the sentence.
    """
    clusters = []
    for public_label, label in sorted(enumerate(set(labels)), reverse=True):
        c = {}
        indices = [i for i, x in enumerate(labels) if x == label]
        if len(indices) == 0:
            continue

        c['size'] = len(indices)
        c['paragraphs'] = [paragraphs[i] for i in indices]
        c['positions'] = indices
        c['features'] = [X[i] for i in indices]

        clusters.append(c)

    # Assuming that sentences talking about the same topic will be close
    # together in the text, we can sort the clusters by average
    # position of the sentences.
    def mean(L):
        return float(sum(L) / max(len(L), 1))

    clusters = sorted(clusters, key=lambda c: mean(c['positions']),
                      reverse=True)
    return clusters


def tfidf(sentences):
    """
    Turn the sentences into a matrix, using the Tf-Idf method.
    """
    stop_words = get_stopwords()
    sentences_processed = [prepare_paragraph(p) for p in sentences]

    min_df = 1 if len(sentences) <= 5 else 3
    max_df = 1 if len(sentences) <= 2 else 0.8
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=None,
                                 ngram_range=(1, 5), norm='l2',
                                 sublinear_tf=True, min_df=min_df,
                                 max_df=max_df)
    features = vectorizer.fit_transform(sentences_processed)
    return features


def plot_clusters(features, labels):
    """
    Plots the "sentences" as "points" in 2D space. The "distances" are
    the sentences "similarity", the closer together, the more similar.

    Useful for debugging the clustering.
    """
    embedding = TSNE(2, init='pca')
    d2 = embedding.fit_transform(features)

    xs = [x[0] for x in d2]
    ys = [x[1] for x in d2]

    pyplot.scatter(xs, ys, c=labels, s=2, cmap='gist_ncar')
    pyplot.axis('off')
    pyplot.savefig('website.png', format='png', dpi=300, bbox_inches='tight')


def clusterize(sentences):
    """
    Splits the sentences into several "groups", or "clusters".
    The goal is to have more "similar" sentences within one cluster.
    """
    X = get_features_lsa(sentences)
    n_clusters = best_number_clusters(X)
    print(time.ctime(), 'Best number of clusters: {}'.format(n_clusters))
    predictor = get_predictor(X, n_clusters)
    labels = predictor.fit_predict(X)

    plot_clusters(X, labels)
    print(time.ctime(), 'Silhouette score:       {:.4f}'.format(silhouette_score(X, labels)))

    return organize_clusters(sentences, labels, X)


def main():
    if len(sys.argv) < 2:
        print('Usage: ./download-website.py <url or file>')
        sys.exit(1)

    if sys.argv[1].startswith('http'):
        html = download(sys.argv[1])
    else:
        html = load_from_disk(sys.argv[1])

    paragraphs = find_paragraphs(html)

    sentences = tokenize_sentences(' '.join(paragraphs))
    if len(sentences) == 0:
        print('The page doesn\'t contain enough meaningful text.')
        sys.exit(1)

    clusters = clusterize(sentences)
    print(time.ctime(), 'Cluster sizes:', [c['size'] for c in clusters])

    # First result-paragraph is a summary of the first webpage-paragraph.
    summary = summarize(paragraphs[0])
    summary = summary[:INTRODUCTION_SUMMARY_LENGTH]
    print('Introduction')
    print('============')
    print(' '.join(summary))
    print()

    # Second result-paragraph is a summary of the whole webpage
    summary = summarize(' '.join(sentences))
    summary = summary[:PAGE_SUMMARY_LENGTH]
    print('Summary')
    print('=======')
    print(' '.join(summary))
    print()

    # Subsequent result-paragraphs operate on sentences.
    print('Additional points')
    print('=================')
    for c in clusters:
        summary = summarize(c['paragraphs'], c['features'])
        summary = summary[:TOPIC_SUMMARY_LENGTH]
        print(' '.join(summary))
        print()


if __name__ == '__main__':
    main()
