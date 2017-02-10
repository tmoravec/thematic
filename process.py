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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

PAGE = 'psychologytoday'
SINCE = '2000-01-01T00:00:00+0000'
N_CLUSTERS = 25


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
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=30000, ngram_range=(1, 2), norm='l2', sublinear_tf=True)
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

        # Skip posts older than SINCE
        if 'created_time' in item:
            t = time.strptime(item['created_time'], '%Y-%m-%dT%H:%M:%S%z')
            if t < time.strptime(SINCE, '%Y-%m-%dT%H:%M:%S%z'):
                continue

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
    pct_rstd = std / float(mean) * 100

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
    vectorizer = CountVectorizer(stop_words=stop_words, max_features=10, ngram_range=(1, 2))
    features = vectorizer.fit_transform(corpus)
    return vectorizer.get_feature_names()


def print_clusters(clusters, use_json=True):
    if use_json:
        template = \
'''
        {{
            "number": {number},
            "important": {important},
            "common": {common},
            "messages": {messages},
            "likes_avg": {likes_avg:.0f},
            "likes_stdev": {likes_stdev:.0f},
            "comments_avg": {comments_avg:.0f},
            "comments_stdev": {comments_stdev:.0f},
            "shares_avg": {shares_avg:.0f},
            "shares_stdev": {shares_stdev:.0f}
        }} '''

        ret = \
'''
{{
    "pagename": {pagename},
    "clusters":
    [
'''.format(pagename='"{}"'.format(PAGE))

        formatted_clusters = []
        for cluster, items in clusters.items():
            important = ['"{}"'.format(i) for i in highest_number_items(items['features'])]
            important = '[{}]'.format(','.join(important))

            common = ['"{}"'.format(i) for i in count_vectorize([' '.join(items['messages'])])]
            common = '[{}]'.format(','.join(common))

            messages = [json.dumps(i) for i in items['messages']]
            messages = '[{}]'.format(','.join(messages))

            c = template.format(
                number=cluster,
                important=important,
                common=common,
                messages=messages,
                likes_avg=list_stats(items['likes'])[0],
                likes_stdev=list_stats(items['likes'])[1],
                comments_avg=list_stats(items['comments'])[0],
                comments_stdev=list_stats(items['comments'])[1],
                shares_avg=list_stats(items['shares'])[0],
                shares_stdev=list_stats(items['shares'])[1]
            )
            formatted_clusters.append(c)

        ret += ',\n'.join(formatted_clusters)
        ret += \
'''
    ]
}
'''
        with open('www/clusters.json', 'w') as f:
            f.write(ret)


    else:
        for cluster, items in clusters.items():
            print('\nCluster ', cluster)
            print('Messages: {}'.format(len(items['messages'])))
            print('Likes:    {:.0f} +- {:.0f} ({:.0f}%)'.format(*list_stats(items['likes'])))
            print('Comments: {:.0f} +- {:.0f} ({:.0f}%)'.format(*list_stats(items['comments'])))
            print('Shares:   {:.0f} +- {:.0f} ({:.0f}%)'.format(*list_stats(items['shares'])))
            print('Important features: ', highest_number_items(items['features']))
            print('Common words:       ', count_vectorize([' '.join(items['messages'])]))

            try:
                for i in range(3):
                    print('Message {}: {}'.format(i, items['messages'][i]))
            except IndexError:
                # There were less than 5 messages in this group.
                pass


def most_important_features(tf, vectorizer, max_count=10, threshold=0.3):
    """
    Selects highest value features from the corpus obtained with Tfidf.
    """
    highest = sorted(tf, reverse=True)

    words = []
    for i, n in enumerate(highest):
        if i > max_count - 1:
            break
        if n < threshold:
            break

        index = np.where(tf == n)[0][0]
        word = vectorizer.get_feature_names()[index]
        words.append([word, n])

    return words


def text_clustering(raw_data):
    likes, comments, shares, tf, msgs, vectorizer = text_features(raw_data)

    svd_components = 200
    print(time.ctime(), 'Generating {} LSA components and normalizing'.format(svd_components))
    svd = TruncatedSVD(svd_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(tf)

    predictor = AgglomerativeClustering(n_clusters=N_CLUSTERS, connectivity=None, linkage='ward', affinity='euclidean')
    print(time.ctime(), 'Starting to fit.')
    labels = predictor.fit_predict(X)

    print(time.ctime(), 'Generating the clusters with values.')
    # {<cluster number>: {'likes': [...], 'comments': [...], 'messages': [...]}}
    clustered = {}
    tf_array = tf.toarray()
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
    print('\n\n')
    print('Messages: ', msgs_counts)
    print('Messages: {:.0f} +- {:.0f} ({:.0f}%)'.format(*list_stats(msgs_counts)))
    print('Labels:      ', len(set(labels)), labels)
    print(time.ctime(), 'Starting to score.')
    print('Silhouette score: ', silhouette_score(X, labels, metric='euclidean'))


def main():
    print(time.ctime(), 'Loading data.')
    raw_data = load_data()
    text_clustering(raw_data)



if __name__ == '__main__':
    main()
