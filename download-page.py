#!/usr/bin/env python3

import sys
import json
import requests
import time
from pprint import pprint
import pickle


TOKEN = 'EAACEdEose0cBAJkERkG5Jy3plIGYBYqTMH6Oi1mpZBWimoGxm2qWHljZBQpNjsywCdX2ALqkSYBLetbmaDXW2uZBKslQ3JGMDn0ZBPPmsm8H30vKiVZCt32pme1VQXFUytpaPqdQkMLwaITTfZAVReQ0pkY8Cm60W3GEhZCZBZBcGJLg6wmULBW4FfTboGBUzrUoZD'
URL = 'https://graph.facebook.com/v2.9/{}?fields={}&access_token=' + TOKEN


def get_page(url, field_name):
    print('Getting page {}'.format(url))
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
        pagename = sys.argv[1]
        if pagename.endswith('.pkl'):
            pagename = pagename.split('.pkl')[0]
    except IndexError:
        print('Usage: ./download-page.py <pagename>')
        sys.exit(1)

    fan_count = get_single_value(URL.format(pagename, 'fan_count'), 'fan_count')
    name = get_single_value(URL.format(pagename, 'name'), 'name')

    queries = {
               'posts': 'posts{message,link,likes.limit(0).summary(true),comments.limit(0).summary(true),shares,description,name,created_time}',
               'videos': 'videos',
               'photos': 'photos',
               }

    results = {
               'posts': [],
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
