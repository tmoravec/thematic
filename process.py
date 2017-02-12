#!/usr/bin/env python3

from collections import OrderedDict
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

PAGE = 'psychologytoday'
SINCE = '2000-01-01T00:00:00+0000'
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


def global_stats(likes, comments, shares, messages):
    likes_stats =    list_stats(likes)
    comments_stats = list_stats(comments)
    shares_stats =   list_stats(shares)

    return {
            'likes_avg':      likes_stats[0],
            'likes_stdev':    likes_stats[1],
            'comments_avg':   comments_stats[0],
            'comments_stdev': comments_stats[1],
            'shares_avg':     shares_stats[0],
            'shares_stdev':   shares_stats[1],
            'messages':       len(messages),
           }

def print_clusters(clusters, g_stats, use_json=True):
    if use_json:
        template = \
'''
        {{
            "number": {number},
            "important": {important},
            "common": {common},
            "messages": {messages},
            "dates_start": {dates_start},
            "dates_end": {dates_end},
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
    "pagename": "{pagename}",
    "globalstats":
    {{
        "messages": {messages},
        "likes_avg": {likes_avg:.0f},
        "likes_stdev": {likes_stdev:.0f},
        "comments_avg": {comments_avg:.0f},
        "comments_stdev": {comments_stdev:.0f},
        "shares_avg": {shares_avg:.0f},
        "shares_stdev": {shares_stdev:.0f}
    }},
    "clusters":
    [
'''
        g_stats['pagename'] = PAGE
        ret = ret.format(**g_stats)

        # TODO: Global stats.
        formatted_clusters = []
        for cluster, items in clusters.items():
            important = ['"{}"'.format(i) for i in highest_number_items(items['features'])]
            important = '[{}]'.format(','.join(important))

            common = ['"{}"'.format(i) for i in count_vectorize([' '.join(items['messages'])])]
            common = '[{}]'.format(','.join(common))

            messages = [json.dumps(i) for i in items['messages']]
            messages = ['[{}, {}]'.format(x, y) for x, y in zip(messages, items['dates'])]
            messages = '[{}]'.format(','.join(messages))

            dates_start = np.mean(items['dates']) - np.std(items['dates'])
            dates_end = np.mean(items['dates']) + np.std(items['dates'])

            c = template.format(
                number=cluster + 1,  # Users expect lists starting with 1.
                important=important,
                common=common,
                messages=messages,
                dates_start=dates_start,
                dates_end=dates_end,
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
        with open('www/clusters/{}.json'.format(PAGE), 'w') as f:
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


def best_number_clusters(X):
    # Try to reduce difference between average and biggest component size
    diffs = []
    cluster_sizes = (5, 7, 10, 12, 15, 17, 20, 22, 25)
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


def likes_clustering(raw_data):
    likes, comments, shares, tf, msgs, dates, vectorizer = text_features(raw_data)
    likes_sorted = sorted(enumerate(likes), key=lambda x: x[1], reverse=True)
    indices_sorted = [x[0] for x in likes_sorted]
    cluster_indices = np.array_split(indices_sorted, 25)

    tf_array = tf.toarray()
    clustered = {}
    for i, cluster in enumerate(cluster_indices):
        clustered[i] = {
             'likes': [],
             'comments': [],
             'messages': [],
             'dates': [],
             'shares': [],
             'features': [],
            }
        for j in cluster:
            clustered[i]['likes'].append(likes[j])
            clustered[i]['comments'].append(comments[j])
            clustered[i]['messages'].append(msgs[j])
            clustered[i]['dates'].append(dates[j])
            clustered[i]['shares'].append(shares[j])
            clustered[i]['features'].append(most_important_features(tf_array[j], vectorizer))

    g_stats = global_stats(likes, comments, shares, msgs)
    print_clusters(clustered, g_stats)


def text_clustering(raw_data):
    likes, comments, shares, tf, msgs, dates, vectorizer = text_features(raw_data)

    svd_components = 200
    print(time.ctime(), 'Generating {} LSA components and normalizing'.format(svd_components))
    svd = TruncatedSVD(svd_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(tf)

    print(time.ctime(), 'Trying to determine best number of clusters.')
    n_clusters = best_number_clusters(X)
    print(time.ctime(), 'Best number of clusters: {}'.format(n_clusters))

    predictor = AgglomerativeClustering(n_clusters=n_clusters, connectivity=None, linkage='ward', affinity='euclidean')
    print(time.ctime(), 'Starting to fit.')
    labels = predictor.fit_predict(X)

    # TODO: This is very slow and probably unnecessary.
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
                            'dates': [],
                            'shares': [],
                            'features': [],
                           }

        clustered[r]['likes'].append(likes[i])
        clustered[r]['comments'].append(comments[i])
        clustered[r]['messages'].append(msgs[i])
        clustered[r]['shares'].append(shares[i])
        clustered[r]['dates'].append(dates[i])

        # Certain value means different things in different messages.
        # Less most important features suggest better defined cluster? Or just a smaller cluster?
        clustered[r]['features'].append(most_important_features(tf_array[i], vectorizer))

    clustered = OrderedDict(sorted(clustered.items()))
    print(time.ctime(), 'Printing...')
    g_stats = global_stats(likes, comments, shares, msgs)
    print_clusters(clustered, g_stats)

    msgs_counts = [len(x['messages']) for x in clustered.values()]
    print('\n\n')
    print('Messages:               ', msgs_counts)
    print('Messages:                {:.0f} +- {:.0f} ({:.0f}%)'.format(*list_stats(msgs_counts)))
    print('Labels:                 ', len(set(labels)), labels)
    print(time.ctime(), 'Starting to score.')
    print('Calinski harabaz score: ', calinski_harabaz_score(X, labels))
    print('Silhouette score:       ', silhouette_score(X, labels))


def main():
    global PAGE
    print(time.ctime(), 'Loading data.')

    try:
        PAGE = sys.argv[1].split('.pkl')[0]
    except IndexError:
        pass

    raw_data = load_data()
    text_clustering(raw_data)
    #likes_clustering(raw_data)



if __name__ == '__main__':
    main()
