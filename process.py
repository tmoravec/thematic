#!/usr/bin/env python3

import copy
import json
import sys
import pickle
import math

import matplotlib
# Use a backend that doesn't require X server.
matplotlib.use('Agg')
from matplotlib import pyplot

import numpy as np
import networkx as nx
import string
import time
from collections import OrderedDict

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import pairwise_distances

import gensim

SINCE = '2012-01-01T00:00:00+0000'
OUTPUT_DIRECTORY = 'clusters'
SMALL_PAGE = False


def plot_2_arrays(a1, a2):
    pyplot.scatter(a1, a2)
    pyplot.show()
    sys.exit()


def plot_clusters(features, labels, pagename):
    embedding = TSNE(2, init='pca')
    d2 = embedding.fit_transform(features)

    xs = [x[0] for x in d2]
    ys = [x[1] for x in d2]

    pyplot.scatter(xs, ys, c=labels, s=2, cmap='gist_ncar')
    #pyplot.xlabel('x')
    #pyplot.ylabel('y', rotation='horizontal')
    pyplot.axis('off')
    pyplot.savefig('{}/{}.png'.format(OUTPUT_DIRECTORY, pagename),
                   format='png', dpi=300, bbox_inches='tight')
    #pyplot.clf()


def load_data(pagename):
    with open(pagename + '.pkl', 'rb') as f:
        return pickle.load(f)


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


def process_message(message):
    translator = str.maketrans({key: None for key in string.punctuation})
    message = message.translate(translator)

    message = message.replace('\n', ' ')
    message = message.replace('  ', ' ')
    words = message.split()

    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(x) for x in words]

    no_numbers = []
    for w in stems:
        if w.isalpha():
            no_numbers.append(w)

    return ' '.join(no_numbers)


def vectorize(messages):
    global SMALL_PAGE

    stop_words = get_stopwords()
    print(time.ctime(), 'Starting to vectorize.')

    # sublinear_tf recommended by TruncatedSVD documentation for use with
    # TruncatedSVD
    if len(messages) < 200:
        print(time.ctime(), len(messages), 'messages. Small page.')
        SMALL_PAGE = True

    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=None,
                                 strip_accents=None, ngram_range=(1, 5),
                                 norm='l2', sublinear_tf=True,
                                 min_df=3, max_df=0.7)
    features = vectorizer.fit_transform(messages)
    if features.shape[1] < 200:
        print(time.ctime(), features.shape[2], 'tfidf features. Small page.')
        SMALL_PAGE = True

    print(time.ctime(), 'Tfidf ignores {} terms.'.format(len(vectorizer.stop_words_)))
    print(time.ctime(), 'Tfidf matrix shape:', features.shape)

    return features, vectorizer


def append_dot(text):
    if (len(text) > 2 and
            not (text[-1] in string.punctuation or
                 text[-2] in string.punctuation)):
        text += '.'
    return text


def text_features(raw_data):
    texts = []
    unique_messages = set()
    messages = []
    shares = []
    likes = []
    comments = []
    dates = []
    for item in raw_data['posts']:

        # Skip posts older than SINCE
        date = time.strptime(item['created_time'], '%Y-%m-%dT%H:%M:%S%z')
        if date < time.strptime(SINCE, '%Y-%m-%dT%H:%M:%S%z'):
            continue
        date = int(time.mktime(date))  # to Unix timestamp

        if ('message' in item and 'shares' in item and 'likes' in item and
                'comments' in item):
            if item['message'] in unique_messages:
                continue

            unique_messages.add(item['message'])
            text = append_dot(item['message'])
            if 'name' in item:
                text += ' ' + append_dot(item['name'])
            if 'description' in item:
                text += ' ' + append_dot(item['description'])
            texts.append(text)
            messages.append(process_message(text))

            shares.append(item['shares']['count'])
            likes.append(item['likes']['summary']['total_count'])
            comments.append(item['comments']['summary']['total_count'])
            dates.append(date)


    features, vectorizer = vectorize(messages)

    # TODO: Wrap the return values to something like Features object...
    return (np.array(likes), np.array(comments), np.array(shares),
            features, texts, dates, vectorizer)


def list_stats(L):
    mean = np.mean(L)
    std = np.std(L)
    if std != 0:
        pct_rstd = std / float(mean) * 100
    else:
        pct_rstd = 0

    return mean, std, pct_rstd


def count_vectorize(corpus, count=10):
    stop_words = get_stopwords()
    messages = []
    for message in corpus:
        messages.append(process_message(message))

    vectorizer = CountVectorizer(stop_words=stop_words,
                                 strip_accents=None, max_features=count,
                                 ngram_range=(1, 5), max_df=0.7)
    features = vectorizer.fit_transform(messages)
    return vectorizer.get_feature_names()


def best_number_clusters(X):
    # Try to reduce difference between average and biggest component size
    # In some cases this works great, in some cases maximizing silhouette_score
    # would work better.
    scores = []

    cluster_sizes = range(15, 41, 1)
    if SMALL_PAGE:
        cluster_sizes = range(10, 20, 1)

    try:
        for n in cluster_sizes:
            knn_graph = kneighbors_graph(X, 20, include_self=False, n_jobs=-1)
            predictor = AgglomerativeClustering(n_clusters=n,
                                                connectivity=knn_graph,
                                                linkage='ward',
                                                affinity='euclidean')
            labels = predictor.fit_predict(X)

            scores.append(silhouette_score(X, labels))
            print(time.ctime(), '{} clusters: silhouette_score: {:.4f}'.format(
                n, silhouette_score(X, labels)))
    except KeyboardInterrupt:
        pass

    index = scores.index(max(scores))
    size = cluster_sizes[index]

    return size


def clusterize(labels, likes, comments, shares, messages, tf, dates):
    clusters = []
    for public_label, label in sorted(enumerate(set(labels)), reverse=True):
        c = {}
        indices = [i for i, x in enumerate(labels) if x == label]
        c['size'] = len(indices)
        c['likes'] = [likes[i] for i in indices]
        c['comments'] = [comments[i] for i in indices]
        c['shares'] = [shares[i] for i in indices]
        c['messages'] = [messages[i] for i in indices]
        c['tf'] = [tf[i] for i in indices]
        c['dates'] = [dates[i] for i in indices]

        clusters.append(c)

    return clusters


def common_words_distance(a, b):
    common_words_count = len(np.intersect1d(np.nonzero(a[0]), np.nonzero(b[0])))
    lens = math.log(a.shape[1]) + math.log(b.shape[1])
    if lens:
        return common_words_count / lens
    else:
        return 0


def common_words_similarity(X):
    pass



def summarize(messages):
    # Shouldn't we use the global tfidf/LSA values?
    # No. We'd need to operate on the whole posts, which
    # makes the summary much less useful.
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(' '.join(messages))
    sentences = set(sentences)

    lower_sentences = [process_message(s) for s in sentences]
    stop_words = get_stopwords()
    bow = CountVectorizer(stop_words=stop_words, strip_accents=None,
                          binary=True, ngram_range=(1, 5),
                          min_df=3, max_df=0.7).fit_transform(lower_sentences)
    similarity_graph = pairwise_distances(bow, metric=common_words_distance,
                                          n_jobs=-1)

    nx_graph = nx.from_numpy_matrix(similarity_graph)
    scores = nx.pagerank_numpy(nx_graph)
    most_characteristic = sorted(((scores[i], s) for i, s in enumerate(sentences)),
                                 reverse=True)

    if len(most_characteristic) > 50:
        most_characteristic = most_characteristic[:50]
    return [x[1] for x in most_characteristic]


def keywords(text):
    try:
        kw = gensim.summarization.keywords(text, words=10, split=True)
    except IndexError:
        return []

    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(x) for x in kw]

    stopwords = get_stopwords()
    blacklist = {
                 'new',
                 'whi',
                 'realli',
                 'timelin',
                }
    stopwords |= blacklist
    words = [x for x in stems if x not in stopwords]

    # Hack OrderedDict to work as an ordered set.
    od = OrderedDict()
    for w in words:
        od[w] = 0

    return list(od.keys())


def print_clusters(labels, likes, comments, shares, messages, tf, dates,
                   vectorizer, orig_size, pagename, fan_count,
                   page_name_displayed):

    # Break messages by sentences.
    # Perform LSA on them.
    # Return them back to their clusters, with the LSA features.
    # Perform the summarization on this.

    globalstats = {
                   'fan_count': fan_count,
                   'messages': orig_size,
                   'likes_avg': int(np.mean(likes)),
                   'likes_stdev': int(np.std(likes)),
                   'comments_avg': int(np.mean(comments)),
                   'comments_stdev': int(np.std(comments)),
                   'shares_avg': int(np.mean(shares)),
                   'shares_stdev': int(np.std(shares)),
                   'graph_uri': 'clusters/' + pagename + '.png'
                  }


    clusters = clusterize(labels, likes, comments, shares, messages, tf, dates)
    clusters = sorted(clusters, key=lambda c: int(np.mean(c['shares'])),
                      reverse=True)

    clusters_print = []
    for i, c in enumerate(clusters):
        print(time.ctime(), 'Printing cluster {}'.format(i))
        likes_avg = int(np.mean(c['likes']))
        likes_stdev = int(np.std(c['likes']))
        comments_avg = int(np.mean(c['comments']))
        comments_stdev = int(np.std(c['comments']))
        shares_avg = int(np.mean(c['shares']))
        shares_stdev = int(np.std(c['shares']))

        summary = summarize(c['messages'])

        common = 0  # count_vectorize(c['messages'], 10)
        important = keywords(' '.join(summary))
        summary = summary[:5]

        dates_start = int(np.mean(c['dates']) - np.std(c['dates']))
        dates_end = int(np.mean(c['dates']) + np.std(c['dates']))

        cluster = {
                   'number': int(i) + 1,  # People expect 1-indexed arrays.
                   'important': important,
                   'summary': summary,
                   'common': common,
                   'messages': list(zip(c['messages'], c['dates'])),
                   'dates_start': dates_start,
                   'dates_end': dates_end,
                   'likes_avg': likes_avg,
                   'likes_stdev': likes_stdev,
                   'comments_avg': comments_avg,
                   'comments_stdev': comments_stdev,
                   'shares_avg': shares_avg,
                   'shares_stdev': shares_stdev,
                  }
        clusters_print.append(cluster)

    result = {
              'pagename': pagename,
              'page_name_displayed': page_name_displayed,
              'globalstats': globalstats,
              'clusters': clusters_print,
             }

    with open('{}/{}.json'.format(OUTPUT_DIRECTORY, pagename), 'w') as f:
        json.dump(result, f, indent=2)


def get_features_lsa(tf):
    svd_components = 100
    if SMALL_PAGE:
        svd_components = 100

    print(time.ctime(), 'Generating {} LSA components and normalizing'.format(svd_components))
    svd = TruncatedSVD(svd_components, algorithm='arpack', tol=0)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(tf)
    return X


def find_largest_cluster(labels):
    counts = np.bincount(labels)
    largest_cluster_size = 0
    largest_cluster_index = 0
    for i, c in enumerate(counts):
        if c > largest_cluster_size:
            largest_cluster_size = c
            largest_cluster_index = i

    return largest_cluster_index


def text_clustering(raw_data, pagename):
    fan_count = raw_data['fan_count']
    page_name_displayed = raw_data['name']
    likes, comments, shares, tf, messages, dates, vectorizer = text_features(raw_data)
    orig_size = len(messages)
    X = get_features_lsa(tf)

    print(time.ctime(), 'Trying to determine best number of clusters.')
    n_clusters = best_number_clusters(X)
    print(time.ctime(), 'Best number of clusters: {}'.format(n_clusters))

    knn_graph = kneighbors_graph(X, 20, include_self=False, n_jobs=-1)
    predictor = AgglomerativeClustering(n_clusters=n_clusters,
                                        connectivity=knn_graph,
                                        linkage='ward',
                                        affinity='euclidean')
    print(time.ctime(), 'Starting to fit.')
    labels = predictor.fit_predict(X)
    print(time.ctime(), 'Cluster sizes:', len(np.bincount(labels)),
          np.bincount(labels))

    print(time.ctime(), 'Calinski harabaz score: {:.4f}'.format(calinski_harabaz_score(X, labels)))
    print(time.ctime(), 'Silhouette score:       {:.4f}'.format(silhouette_score(X, labels)))

    print(time.ctime(), 'Drawing.')
    plot_clusters(X, labels, pagename)

    print(time.ctime(), 'Printing.')
    print_clusters(labels, likes, comments, shares, messages, tf, dates,
                   vectorizer, orig_size, pagename, fan_count,
                   page_name_displayed)


def main():
    print(time.ctime(), 'Loading data.')

    try:
        pagename = sys.argv[1].split('.pkl')[0]
    except IndexError:
        pagename = 'psychologytoday'

    raw_data = load_data(pagename)
    text_clustering(raw_data, pagename)



if __name__ == '__main__':
    main()
