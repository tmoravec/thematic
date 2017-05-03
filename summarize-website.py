#!/usr/bin/env python3

import sys
import time
import re
import string

from bs4 import BeautifulSoup
import requests

import numpy as np

import networkx as nx

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

INTRODUCTION_SUMMARY_LENGTH=2
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
    if '{' in text:
        return False

    if len(text) < 10:
        return False

    return True


def tokenize_sentences(text):
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(text)
    return list(set(sentences))


def process_paragraph_rough(text):
    text = re.sub('\[[^\]]*\]', '', text)
    text = re.sub('(č\.)\s+', 'č.', text)
    #text = re.sub('(s\.)\s+', 's.', text)
    text = re.sub('(p\.)\s+', 'p.', text)
    text = re.sub('(mj\.)\s+', 'mj.', text)
    text = re.sub('(tzv\.)\s+', 'tzv.', text)
    text = re.sub('(např\.)\s+', 'např.', text)
    text = re.sub('(\s+\d+)\s+', '\1', text)
    return text


def process_paragraph_fine(text):
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


def summarize(text, X=None):
    # Accepts either text itself, or a list of sentences with corresponding
    # LSA features.

    if not X:
        sentences = tokenize_sentences(text)
        X = get_features_lsa(sentences)
    else:
        sentences = text

    if len(sentences) > 2000:
        return ''

    similarity_graph = linear_kernel(X)

    nx_graph = nx.from_numpy_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-5)
    most_characteristic = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    if len(most_characteristic) > 10:
        most_characteristic = most_characteristic[:10]
    return [x[1] for x in most_characteristic]


def find_paragraphs(html):
    soup = BeautifulSoup(html, 'html5lib')
    paragraphs = soup.find_all('p')

    cleaned = []
    for p in paragraphs:
        text = p.get_text().strip()
        if validate_paragraph(text):
            text = process_paragraph_rough(text)
            cleaned.append(text)

    if not cleaned:
        # No <p> tags.
        cleaned = [soup.get_text()]
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
    features = tfidf(sentences)
    min_dimensions = min(features.shape)
    if min_dimensions > 400:
        svd_components = 120
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
    return X


def best_number_clusters(X):
    # Try to reduce difference between average and biggest component size
    # In some cases this works great, in some cases maximizing silhouette_score
    # would work better.
    scores = []

    print(time.ctime(), 'best_number_clusters. X.shape:', X.shape)
    cluster_sizes = range(3, 9, 1)
    if max(X.shape) > 1000:
        cluster_sizes = range(5, 16, 1)

    try:
        for n in cluster_sizes:
            predictor = AgglomerativeClustering(n_clusters=n, connectivity=None, linkage='ward', affinity='euclidean')
            labels = predictor.fit_predict(X)

            scores.append(silhouette_score(X, labels))
    except KeyboardInterrupt:
        pass

    index = scores.index(max(scores))
    size = cluster_sizes[index]

    return size


def find_largest_cluster(labels):
    counts = np.bincount(labels)
    largest_cluster_size = 0
    largest_cluster_index = 0
    for i, c in enumerate(counts):
        if c > largest_cluster_size:
            largest_cluster_size = c
            largest_cluster_index = i

    return largest_cluster_index


def organize_clusters(paragraphs, labels, X):
    # Order the clusters by average position of the sentence.
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
    clusters = sorted(clusters, key=lambda c: int(np.mean(c['positions'])))
    return clusters


def tfidf(sentences):
    stop_words = get_stopwords()
    sentences_processed = [process_paragraph_fine(p) for p in sentences]

    min_df = 2 if len(sentences) <= 5 else 3
    max_df = 1 if len(sentences) <= 2 else 0.8
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=None,
                                 ngram_range=(1, 5), norm='l2',
                                 sublinear_tf=True, min_df=min_df, max_df=max_df)
    features = vectorizer.fit_transform(sentences_processed)
    return features


def clusterize(sentences):
    X = get_features_lsa(sentences)
    n_clusters = best_number_clusters(X)
    print(time.ctime(), 'Best number of clusters: {}'.format(n_clusters))

    predictor = AgglomerativeClustering(n_clusters=n_clusters, connectivity=None, linkage='ward', affinity='euclidean')
    labels = predictor.fit_predict(X)
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
    if len(summary) > INTRODUCTION_SUMMARY_LENGTH:
        summary = summary[:INTRODUCTION_SUMMARY_LENGTH]
    print(' '.join(summary))
    print()

    # Second result-paragraph is a summary of the whole webpage
    summary = summarize(' '.join(sentences))
    if len(summary) > PAGE_SUMMARY_LENGTH:
        summary = summary[:PAGE_SUMMARY_LENGTH]
    print(' '.join(summary))
    print()

    # Subsequent result-paragraphs operate on sentences.
    for c in clusters:
        summary = summarize(c['paragraphs'], c['features'])
        if len(summary) > TOPIC_SUMMARY_LENGTH:
            summary = summary[:TOPIC_SUMMARY_LENGTH]
        print(' '.join(summary))
        print()


if __name__ == '__main__':
    main()
