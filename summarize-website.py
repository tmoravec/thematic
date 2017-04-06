#!/usr/bin/env python3

from bs4 import BeautifulSoup
import requests
import sys
import time
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import networkx as nx
import string
from nltk.corpus import stopwords


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


def validate_paragraph(text):
    if '{' in text:
        return False

    if len(text) < 10:
        return False

    return True


def process_paragraph_rough(text):
    text = re.sub('\[[^\]]*\]', '', text)
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


def summarize(text):
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(text)
    sentences = set(sentences)

    lower_sentences = [s.lower() for s in sentences]
    tfidf = TfidfVectorizer().fit_transform(lower_sentences)
    similarity_graph = linear_kernel(tfidf)

    nx_graph = nx.from_numpy_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    most_characteristic = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    if len(most_characteristic) > 50:
        most_characteristic = most_characteristic[:50]
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


def get_features_lsa(tf):
    svd_components = min(tf.shape) - 1
    if svd_components > 100:
        svd_components = 100

    print(time.ctime(), 'Generating {} LSA components and normalizing'.format(svd_components))
    svd = TruncatedSVD(svd_components, algorithm='arpack', tol=0)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(tf)
    return X


def best_number_clusters(X):
    # Try to reduce difference between average and biggest component size
    # In some cases this works great, in some cases maximizing silhouette_score
    # would work better.
    scores = []

    cluster_sizes = range(3, 7, 1)

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


def organize_clusters(paragraphs, labels):
    clusters = []
    for public_label, label in sorted(enumerate(set(labels)), reverse=True):
        c = {}
        indices = [i for i, x in enumerate(labels) if x == label]
        c['size'] = len(indices)
        c['paragraphs'] = [paragraphs[i] for i in indices]

        clusters.append(c)

    return clusters


def clusterize(paragraphs):
    stop_words = get_stopwords()
    paragraphs_processed = [process_paragraph_fine(p) for p in paragraphs]
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=None,
                                 ngram_range=(1, 5), norm='l2',
                                 sublinear_tf=True, min_df=2, max_df=0.7)
    features = vectorizer.fit_transform(paragraphs_processed)
    print(time.ctime(), 'Tfidf features shape:', features.shape)

    X = get_features_lsa(features)
    n_clusters = best_number_clusters(X)
    print(time.ctime(), 'Best number of clusters: {}'.format(n_clusters))

    predictor = AgglomerativeClustering(n_clusters=n_clusters, connectivity=None, linkage='ward', affinity='euclidean')
    labels = predictor.fit_predict(X)
    print(time.ctime(), 'Silhouette score:       {:.4f}'.format(silhouette_score(X, labels)))

    return organize_clusters(paragraphs, labels)


def main():
    if len(sys.argv) < 2:
        print('Usage: ./download-website.py <url>')

    html = download(sys.argv[1])
    paragraphs = find_paragraphs(html)

    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(' '.join(paragraphs))
    sentences = list(set(sentences))

    clusters = clusterize(sentences)

    summary = summarize(paragraphs[0])

    summary_paragraph_length = 2
    if len(summary) > summary_paragraph_length:
        summary = summary[:summary_paragraph_length]
    print(' '.join(summary))
    print()

    for c in clusters:
        summary = summarize(' '.join(c['paragraphs']))
        if len(summary) > summary_paragraph_length:
            summary = summary[:summary_paragraph_length]
        print(' '.join(summary))
        print()


if __name__ == '__main__':
    main()
