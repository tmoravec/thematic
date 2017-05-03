#!/usr/bin/env python3

import json
import sys
from datetime import date


def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def print_cluster(c):
    print('Topic', c['number'])
    print(len(c['messages']), 'posts')
    print('Most important words: ')
    for w in c['important']:
        print(w, end=', ')
    print()

    print('Average likes:', c['likes_avg'])
    print('Average comments:', c['comments_avg'])
    print('Average shares:', c['shares_avg'])

    dates_start = date.fromtimestamp(c['dates_start'])
    dates_end = date.fromtimestamp(c['dates_end'])

    start = [dates_start.strftime('%Y'), dates_start.strftime('%m')]
    end =   [dates_end.strftime('%Y'), dates_end.strftime('%m')]

    if start[1].startswith('0'):
        start[1] = start[1][1:]
    
    if end[1].startswith('0'):
        end[1] = end[1][1:]

    print('Roughly from {} to {}'.format(start[0] + '/' + start[1],
                                         end[0]   + '/' + end[1]))
    

    print()


def main():
    filename = sys.argv[1]
    data = load_data(filename)

    for c in data['clusters']:
        print_cluster(c)



if __name__ == '__main__':
    main()
