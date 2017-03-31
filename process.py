#!/usr/bin/env python3

import copy
import json
import sys
import pickle
from matplotlib import pyplot
import numpy as np
import string
import time
from collections import OrderedDict

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE

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
    pyplot.savefig('{}/{}.png'.format(OUTPUT_DIRECTORY, pagename), format='png', dpi=300, bbox_inches='tight')
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

    min_df_range = range(10, 1, -1)
    if SMALL_PAGE:
        min_df_range = range(3, 1, -1)

    for i in min_df_range:
        vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=None,
                                     ngram_range=(1, 5), norm='l2',
                                     sublinear_tf=True, min_df=i, max_df=0.7)
        features = vectorizer.fit_transform(messages)
        if features.shape[1] > 200:
            print(time.ctime(), 'min_df:', i)
            break

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
    for item in raw_data['feed']:

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
    return np.array(likes), np.array(comments), np.array(shares), features, texts, dates, vectorizer


def list_stats(L):
    mean = np.mean(L)
    std = np.std(L)
    if std != 0:
        pct_rstd = std / float(mean) * 100
    else:
        pct_rstd = 0

    return mean, std, pct_rstd


def highest_number_items(items, max_count=100):
    """
    From a nested list, with leaves formed of tuples (whatever, number),
    selects up to max_count tuples with highest number, globally.
    """
    all = []
    for i in items:
        all += i
    s = sorted(all, key=lambda x: x[1], reverse=True)
    used = []
    hnitems = [x[0] for x in s[:max_count] if x[0] not in used and (used.append(x[0]) or True)]
    return hnitems[:10]


def count_vectorize(corpus, count=10):
    stop_words = get_stopwords()
    messages = []
    for message in corpus:
        messages.append(process_message(message))

    vectorizer = CountVectorizer(stop_words=stop_words, max_features=count, ngram_range=(1, 5), max_df=0.7)
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
            predictor = AgglomerativeClustering(n_clusters=n, connectivity=None, linkage='ward', affinity='euclidean')
            labels = predictor.fit_predict(X)

            indices_to_keep = remove_noise_clusters(X, labels)
            labelsI = [labels[i] for i in indices_to_keep]
            XI = [X[i] for i in indices_to_keep]

            scores.append(silhouette_score(XI, labelsI))
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


def summarize(messages):
    text = ' '.join(messages)
    summary = gensim.summarization.summarize(text, word_count=70)
    return summary


def keywords(messages):
    text = ' '.join(messages)
    kw = gensim.summarization.keywords(text, words=10, split=True)

    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(x) for x in kw]

    # Hack OrderedDict to work as an ordered set.
    od = OrderedDict()
    for w in stems:
        od[w] = 0

    return list(od.keys())


def print_clusters(labels, likes, comments, shares, messages, tf, dates,
                   vectorizer, orig_size, pagename, fan_count, page_name_displayed):

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
    clusters = sorted(clusters, key=lambda c: int(np.mean(c['shares'])), reverse=True)

    clusters_print = []
    for i, c in enumerate(clusters):
        likes_avg = int(np.mean(c['likes']))
        likes_stdev = int(np.std(c['likes']))
        comments_avg = int(np.mean(c['comments']))
        comments_stdev = int(np.std(c['comments']))
        shares_avg = int(np.mean(c['shares']))
        shares_stdev = int(np.std(c['shares']))

        summary = summarize(c['messages'])

        common = count_vectorize(c['messages'], 10)
        important = keywords(c['messages'])

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
    svd_components = 200
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


def remove_noise_clusters(X, labels):
    # Print scores while removing clusters for start.
    # Return indices to keep.
    labelsI = copy.deepcopy(labels)
    last_indices = range(len(labels))
    last_score = silhouette_score(X, labels)

    i = 0
    while len(labelsI):
        largest_cluster_index = find_largest_cluster(labelsI)
        indices_to_keep = np.where(labelsI != largest_cluster_index)[0]
        XI = np.array([X[x] for x in indices_to_keep])
        labelsI = np.array([labels[x] for x in indices_to_keep])

        #score_ch = calinski_harabaz_score(XI, labelsI)
        score_sil = silhouette_score(XI, labelsI)

        if score_sil < last_score:
            break

        print(time.ctime(), 'Last score: {:.4f}, removed cluster {}, current score: {:.4f}'.format(last_score, i, score_sil))
        last_score = score_sil
        last_indices = indices_to_keep


        i += 1

    return last_indices


def text_clustering(raw_data, pagename):
    fan_count = raw_data['fan_count']
    page_name_displayed = raw_data['name']
    likes, comments, shares, tf, messages, dates, vectorizer = text_features(raw_data)
    orig_size = len(messages)

    X = get_features_lsa(tf)

    print(time.ctime(), 'Trying to determine best number of clusters.')
    n_clusters = best_number_clusters(X)
    print(time.ctime(), 'Best number of clusters: {}'.format(n_clusters))

    predictor = AgglomerativeClustering(n_clusters=n_clusters, connectivity=None, linkage='ward', affinity='euclidean')
    #predictor = KMeans(n_clusters=n_clusters)
    print(time.ctime(), 'Starting to fit.')
    labels = predictor.fit_predict(X)
    print(time.ctime(), 'Cluster sizes:', len(np.bincount(labels)), np.bincount(labels))

    indices_to_keep = remove_noise_clusters(X, labels)

    labels = [labels[i] for i in indices_to_keep]
    X = [X[i] for i in indices_to_keep]
    likes = [likes[i] for i in indices_to_keep]
    comments = [comments[i] for i in indices_to_keep]
    shares = [shares[i] for i in indices_to_keep]
    tf = [tf[i].toarray()[0].tolist() for i in indices_to_keep]
    messages = [messages[i] for i in indices_to_keep]
    dates = [dates[i] for i in indices_to_keep]


    print(time.ctime(), 'Calinski harabaz score: {:.4f}'.format(calinski_harabaz_score(X, labels)))
    print(time.ctime(), 'Silhouette score:       {:.4f}'.format(silhouette_score(X, labels)))

    print(time.ctime(), 'Drawing.')
    plot_clusters(X, labels, pagename)

    print(time.ctime(), 'Printing.')
    print_clusters(labels, likes, comments, shares, messages, tf, dates, vectorizer, orig_size, pagename, fan_count, page_name_displayed)


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
