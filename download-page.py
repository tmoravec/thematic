#!/usr/bin/env python3

import sys
import json
import requests
import time
from pprint import pprint
import pickle


TOKEN = 'EAACEdEose0cBAHH8nZAVHf4bCGtnnii2fwcXS6xQqxOhaXR3ZC3tb1lWck3JxL4zitoueyQUEF81arVvznqNZBDcFPEL6eZBZB27IvJveZBgxQiU0bXIoAAAfmBh2jOaFoJjioidWfS55ZA1ZAR2rHOpQwVyppqTLZCae0HXQK7sGvSVmZB0nTqfqFp1ZB1nZB3KwXcZD'
URL = 'https://graph.facebook.com/v2.8/{}?fields={}&access_token=' + TOKEN


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
    try:
        pagename = sys.argv[1].split('.pkl')[0]
    except IndexError:
        pagename = 'psychologytoday'

    fan_count = get_single_value(URL.format(pagename, 'fan_count'), 'fan_count')
    name = get_single_value(URL.format(pagename, 'name'), 'name')

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
            next_url = URL.format(pagename, v)
            while '' != next_url:
                content, next_url = get_page(next_url, k)
                results[k] += content

                print('Got page {}'.format(i))
                i += 1
                time.sleep(1)
    except KeyboardInterrupt:
        pass

    results['fan_count'] = fan_count
    results['name'] = name

    with open('{}.pkl'.format(pagename), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
