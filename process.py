#!/usr/bin/env python3

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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.base import TransformerMixin
from sklearn.decomposition import KernelPCA
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.svm import SVR
from sklearn.cluster import KMeans, MiniBatchKMeans  # MiniBatchKMeans very fast. Quite evenly distributed.
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture  # Very slow. One huge cluster.
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph



PAGE = 'psychologytoday'
N_CLUSTERS = 20


def plot_2_arrays(a1, a2):
    pyplot.scatter(a1, a2)
    pyplot.show()
    sys.exit()


def load_data():
    with open(PAGE + '.pkl', 'rb') as f:
        return pickle.load(f)


def numeric_features(raw_data):
    i = 0
    distilleries = []
    fan_count = []
    photos = []
    videos = []
    messages = []
    for k, v in raw_data.items():
        distilleries.append(i)
        fan_count.append(v['fan_count'])
        photos.append(len(v['photos']))
        videos.append(len(v['videos']))
        messages.append(len(v['feed']))
        i += 1

    n = np.array([distilleries, fan_count, photos, videos, messages])
    return n


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

    return ' '.join(no_www)


def vectorize(messages):
    stop_words = set(stopwords.words('english'))
    print(time.ctime(), 'Starting to vectorize.')
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=5000, ngram_range=(1, 1), norm='l2')
    features = vectorizer.fit_transform(messages)

    return features, vectorizer


def text_features(raw_data):
    texts = []
    unique_msgs = set()
    messages = []
    shares = []
    likes = []
    comments = []
    for item in raw_data['feed']:
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

    features, vectorizer = vectorize(messages)

    return np.array(likes), np.array(comments), np.array(shares), features, texts, vectorizer


def most_common_words(corpus):
    # Get most "important" words according to Tfidf?
    words = corpus.split()
    stop_words = set(stopwords.words('english'))
    cleaned = [w for w in words if w not in stop_words]
    word_counts = Counter(cleaned)
    ordered = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return ordered[:10]


def list_stats(L):
    mean = np.mean(L)
    std = np.std(L)
    pct_rstd = std / float(mean) * 100

    return mean, std, pct_rstd


def print_clusters(clusters):

    def highest_number_items(items):
        all = []
        for i in items:
            all += i
        s = sorted(all, key=lambda x: x[1], reverse=True)
        used = []
        hnitems = [x[0] for x in s[:100] if x[0] not in used and (used.append(x[0]) or True)]
        return hnitems[:10]

    def count_vectorize(corpus):
        stop_words = set(stopwords.words('english'))
        vectorizer = CountVectorizer(stop_words=stop_words, max_features=10, ngram_range=(1, 1))
        features = vectorizer.fit_transform(corpus)
        return vectorizer.get_feature_names()


    for cluster, items in clusters.items():
        print('\nCluster ', cluster)
        print('Messages: {}'.format(len(items['messages'])))
        print('Likes:    {} +- {} ({}%)'.format(*list_stats(items['likes'])))
        print('Comments: {} +- {} ({}%)'.format(*list_stats(items['comments'])))
        print('Shares:   {} +- {} ({}%)'.format(*list_stats(items['shares'])))
        print('Important features: ', highest_number_items(items['features']))
        print('Common words:       ', count_vectorize([' '.join(items['messages'])]))

        try:
            for i in range(3):
                print('Message {}: {}'.format(i, items['messages'][i]))
        except IndexError:
            # There were less than 5 messages in this group.
            pass


def most_important_features(tf, vectorizer):
    highest = sorted(tf, reverse=True)

    words = []
    for i, n in enumerate(highest):
        if i > 9:
            break
        if n < 0.5:
            break

        index = np.where(tf==n)[0][0]
        word = vectorizer.get_feature_names()[index]
        words.append([word, n])

    return words




def text_clustering(raw_data):
    likes, comments, shares, tf, msgs, vectorizer = text_features(raw_data)
    tf_array = tf.toarray()

    print(time.ctime(), 'Generating the neighbors graph.')
    neighbors = kneighbors_graph(tf_array, 2, include_self=False, n_jobs=4)

    predictor = AgglomerativeClustering(n_clusters=N_CLUSTERS, connectivity=neighbors, linkage='ward', affinity='euclidean')
    #predictor = Birch(threshold=0.9, branching_factor=1000, n_clusters=N_CLUSTERS)  # TODO: Test n_clusters=None
    #predictor = DBSCAN(eps=0.9, min_samples=10, n_jobs=7)
    #predictor = MiniBatchKMeans(n_clusters=N_CLUSTERS, n_init=3)

    print(time.ctime(), 'Starting to fit.')
    labels = predictor.fit_predict(tf_array)

    print(time.ctime(), 'Generating the clusters with values.')
    # {<cluster number>: {'likes': [...], 'comments': [...], 'messages': [...]}}
    clustered = {}
    for i, r in enumerate(labels):
        if r not in clustered:
            clustered[r] = {
                            'likes': [],
                            'comments': [],
                            'messages': [],
                            'shares': [],
                            'features': [],
                           }

        clustered[r]['likes'].append(likes[i])
        clustered[r]['comments'].append(comments[i])
        clustered[r]['messages'].append(msgs[i])
        clustered[r]['shares'].append(shares[i])

        # Certain value means different things in different messages.
        # Less most important features suggest better defined cluster? Or just a smaller cluster?
        clustered[r]['features'].append(most_important_features(tf_array[i], vectorizer))

    print(time.ctime(), 'Printing...')
    print_clusters(clustered)

    msgs_counts = [len(x['messages']) for x in clustered.values()]
    print('Messages: ', msgs_counts)
    print('Messages: {} +- {} ({}%)'.format(*list_stats(msgs_counts)))
    #print('Subclusters: ', len(predictor.subcluster_labels_))
    print('Labels:      ', len(set(labels)), labels)
    print(time.ctime(), 'Starting to score.')
    print('Silhouette score: ', silhouette_score(tf_array, labels, metric='euclidean'))


def main():
    print(time.ctime(), 'Loading data.')
    raw_data = load_data()

    text_clustering(raw_data)



if __name__ == '__main__':
    main()
