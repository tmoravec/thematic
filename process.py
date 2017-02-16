#!/usr/bin/env python3

import json
import sys
import pickle
from matplotlib import pyplot
import numpy as np
import string
import time
from collections import Counter

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity

import gensim

PAGE = 'AkamaiTechnologies'
SINCE = '2012-01-01T00:00:00+0000'
N_CLUSTERS = 10


def plot_2_arrays(a1, a2):
    pyplot.scatter(a1, a2)
    pyplot.show()
    sys.exit()


def plot_clusters(features, labels):
    svd = TruncatedSVD(2)
    d2 = svd.fit_transform(features)

    xs = [x[0] for x in d2]
    ys = [x[1] for x in d2]

    pyplot.scatter(xs, ys, c=labels)
    pyplot.show()
    sys.exit()


def load_data():
    with open(PAGE + '.pkl', 'rb') as f:
        return pickle.load(f)


def process_message(msg):
    translator = str.maketrans({key: None for key in string.punctuation})
    msg = msg.translate(translator)

    msg = msg.replace('\n', ' ')
    msg = msg.replace('  ', ' ')
    words = msg.split()

    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(x) for x in words]

    no_www = []
    for s in stems:
        if not (s.startswith('http') or s.startswith('www')):
            no_www.append(s)

    no_numbers = []
    for w in no_www:
        if w.isalpha():
            no_numbers.append(w)

    return ' '.join(no_numbers)


def vectorize(messages):
    stop_words = set(stopwords.words('english'))
    print(time.ctime(), 'Starting to vectorize.')

    # sublinear_tf recommended by TruncatedSVD documentation for use with
    # TruncatedSVD
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=None, ngram_range=(1, 5), norm='l2', sublinear_tf=True, min_df=10)
    features = vectorizer.fit_transform(messages)
    print(time.ctime(), 'Tfidf ignores {} terms.'.format(len(vectorizer.stop_words_)))

    return features, vectorizer


def text_features(raw_data):
    texts = []
    unique_msgs = set()
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
            if item['message'] in unique_msgs:
                continue

            unique_msgs.add(item['message'])
            text = item['message']
            if 'name' in item:
                text += ' ' + item['name']
            if 'description' in item:
                text += ' ' + item['description']
            texts.append(text)
            messages.append(process_message(text))

            shares.append(item['shares']['count'])
            likes.append(item['likes']['summary']['total_count'])
            comments.append(item['comments']['summary']['total_count'])
            dates.append(date)


    features, vectorizer = vectorize(messages)

    # TODO: Wrap the return values to something like Features object...
    return np.array(likes), np.array(comments), np.array(shares), features, texts, dates, vectorizer


def most_common_words(corpus, n=10):
    """
    Select n most common words from a given corpus.
    """
    words = corpus.split()
    stop_words = set(stopwords.words('english'))
    cleaned = [w for w in words if w not in stop_words]
    word_counts = Counter(cleaned)
    ordered = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return ordered[:n]


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


def count_vectorize(corpus):
    stop_words = set(stopwords.words('english'))
    vectorizer = CountVectorizer(stop_words=stop_words, max_features=10, ngram_range=(1, 5))
    features = vectorizer.fit_transform(corpus)
    return vectorizer.get_feature_names()


def best_number_clusters(X):
    # Try to reduce difference between average and biggest component size
    diffs = []
    cluster_sizes = (5, 10, 15, 20, 25, 30)
    for n in cluster_sizes:
        predictor = AgglomerativeClustering(n_clusters=n, connectivity=None, linkage='ward', affinity='euclidean')
        labels = predictor.fit_predict(X)
        sizes = np.bincount(labels)
        avg_size = np.mean(sizes)
        diffs.append(max(sizes) - avg_size)

    index = diffs.index(min(diffs))
    size = cluster_sizes[index]


    #guess = int(len(X) / 20)
    #size = min([size, guess])
    #size = max([size, 5])

    return size


def most_important_features(tf, vectorizer, count=10):
    highest = sorted(list(np.array(tf).flatten()), reverse=True)
    words = set()
    feature_names = vectorizer.get_feature_names()
    for i in tf:
        for h in highest:
            if h in i:
                word = feature_names[list(i).index(h)]
                words.add(word)
                if len(words) == count:
                    return list(words)

    return list(words)


def print_clusters(labels, likes, comments, shares, messages, tf_array,
                    dates, vectorizer):

    globalstats = {
                   'messages': len(messages),
                   'likes_avg': int(np.mean(likes)),
                   'likes_stdev': int(np.std(likes)),
                   'comments_avg': int(np.mean(comments)),
                   'comments_stdev': int(np.std(comments)),
                   'shares_avg': int(np.mean(shares)),
                   'shares_stdev': int(np.std(shares)),
                  }


    clusters = []
    for label in set(labels):
        indices = [i for i, x in enumerate(labels) if x == label]
        c_likes = [likes[i] for i in indices]
        c_comments = [comments[i] for i in indices]
        c_shares = [shares[i] for i in indices]
        c_messages = [messages[i] for i in indices]
        c_tf = [tf_array[i] for i in indices]
        c_dates = [dates[i] for i in indices]

        likes_avg = int(np.mean(c_likes))
        likes_stdev = int(np.std(c_likes))
        comments_avg = int(np.mean(c_comments))
        comments_stdev = int(np.std(c_comments))
        shares_avg = int(np.mean(c_shares))
        shares_stdev = int(np.std(c_shares))

        important = most_important_features(c_tf, vectorizer)
        common = count_vectorize(c_messages)

        dates_start = int(np.mean(c_dates) - np.std(c_dates))
        dates_end = int(np.mean(c_dates) + np.std(c_dates))

        cluster = {
                   'number': int(label) + 1,  # People expect 1-indexed arrays.
                   'important': important,
                   'common': common,
                   'messages': list(zip(c_messages, c_dates)),
                   'dates_start': dates_start,
                   'dates_end': dates_end,
                   'likes_avg': likes_avg,
                   'likes_stdev': likes_stdev,
                   'comments_avg': comments_avg,
                   'comments_stdev': comments_stdev,
                   'shares_avg': shares_avg,
                   'shares_stdev': shares_stdev,
                  }
        clusters.append(cluster)

    result = {
              'pagename': PAGE,
              'globalstats': globalstats,
              'clusters': clusters,
             }

    with open('www/clusters/{}.json'.format(PAGE), 'w') as f:
        json.dump(result, f, indent=2)


def get_features_lsa(raw_data):
    likes, comments, shares, tf, msgs, dates, vectorizer = text_features(raw_data)

    svd_components = 200
    print(time.ctime(), 'Generating {} LSA components and normalizing'.format(svd_components))
    svd = TruncatedSVD(svd_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(tf)
    print(svd.components_[0])

    return X


def text_clustering(raw_data):
    likes, comments, shares, tf, msgs, dates, vectorizer = text_features(raw_data)

    X = get_features_lsa(raw_data)


    print(time.ctime(), 'Trying to determine best number of clusters.')
    n_clusters = best_number_clusters(X)
    print(time.ctime(), 'Best number of clusters: {}'.format(n_clusters))

    predictor = AgglomerativeClustering(n_clusters=n_clusters, connectivity=None, linkage='ward', affinity='euclidean')
    print(time.ctime(), 'Starting to fit.')
    labels = predictor.fit_predict(X)

    print(time.ctime(), 'Printing.')
    tf_array = tf.toarray()
    print_clusters(labels, likes, comments, shares, msgs, tf_array, dates, vectorizer)

    print('Labels:                 ', len(set(labels)), labels)
    print(time.ctime()              , 'Starting to score.')
    print('Calinski harabaz score: ', calinski_harabaz_score(X, labels))
    print('Silhouette score:       ', silhouette_score(X, labels))


def create_bag_of_centroids(wordlist, word_centroid_map):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype='float32')
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


def get_features_w2v(raw_data):
    likes, comments, shares, tf, msgs, dates, vectorizer = text_features(raw_data)

    msgs = [process_message(m).split() for m in msgs]
    model = gensim.models.Word2Vec(msgs, size=200, window=5, min_count=1, workers=4)
    model.init_sims(True)

    n_clusters = int(model.syn0.shape[0] / 5)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', affinity='euclidean')
    idx = clustering.fit_predict(model.syn0)
    word_centroid_map = dict(zip(model.index2word, idx))

    X = []
    for msg in msgs:
        bag = create_bag_of_centroids(msg, word_centroid_map)
        X.append(bag)

    return X



def main():
    global PAGE
    print(time.ctime(), 'Loading data.')

    try:
        PAGE = sys.argv[1].split('.pkl')[0]
    except IndexError:
        pass

    raw_data = load_data()
    text_clustering(raw_data)



if __name__ == '__main__':
    main()
