#!/usr/bin/env python3

import json
import requests
import time
from pprint import pprint
import pickle


TOKEN = 'EAACEdEose0cBAAE8kCuB6cZBsomcI5WOk4LrZCDwFmVHe3p6MPd5Fo7KOmkcn13zYZAXCLMz6mWDkK0oRRNVVJZB1IP4sZB7thRmcQoeRAGdGYq6ZAcAqJWZBbNZBiwePw6P0AGYOBPYMx0AbFqT4IHHCZBsICd9P5fdOIIhfIt9GZC7mah8eGMDdkeSFGaiu09gIZD'
PAGE = 'SamsungMobile'
URL = 'https://graph.facebook.com/v2.8/' + PAGE + '?fields={}&access_token=' + TOKEN
DIRECTORY = 'pages-' + PAGE
FILE_NAME = DIRECTORY + '/page-{}.json'


def get_page(url, field_name):
    print('Getting page {}'.format(url))
    r = requests.get(url)
    try:
        content = r.json()
    except json.decoder.JSONDecodeError:
        return [], ''

    next_url = ''
    try:
        next_url = content['paging']['next']
    except KeyError:
        try:
            next_url = content[field_name]['paging']['next']
        except KeyError:
            pass

    try:
        ret = content[field_name]['data']
    except KeyError:
        try:
            ret = content['data']
        except KeyError:
            ret = []

    return ret, next_url


def get_single_value(url, name):
    print('Getting page {}'.format(url))

    r = requests.get(url)
    content = r.json()

    return content[name]


def main():
    fan_count = get_single_value(URL.format('fan_count'), 'fan_count')

    queries = {
               'feed': 'feed{message,link,likes.limit(0).summary(true),comments.limit(0).summary(true),shares,description,name,created_time}',
               'videos': 'videos',
               'photos': 'photos',
               }

    results = {
               'feed': [],
               'videos': [],
               'photos': []
              }

    i = 0
    try:
        for k, v in queries.items():
            next_url = URL.format(v)
            while '' != next_url:
                content, next_url = get_page(next_url, k)
                results[k] += content

                print('Got page {}'.format(i))
                i += 1
                time.sleep(1)
    except KeyboardInterrupt:
        pass

    results['fan_count'] = fan_count

    with open('{}.pkl'.format(PAGE), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
