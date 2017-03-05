#!/usr/bin/env python3

from matplotlib import pyplot
import numpy as np
from sklearn.cluster import KMeans

x1 = np.random.normal(loc=1, size=(20,))
y1 = np.random.normal(loc=1, size=(20,))

x2 = np.random.normal(loc=5, size=(20,))
y2 = np.random.normal(loc=5, size=(20,))

x3 = np.random.normal(loc=10, size=(20,))
y3 = np.random.normal(loc=10, size=(20,))

x = np.concatenate([x1, x3, x2])
y = np.concatenate([y2, y1, y3])



X = list(zip(x, y))

kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)

pyplot.figure(1, tight_layout=True)
pyplot.subplot(211)
pyplot.scatter(x, y, c='black')
pyplot.xlabel('x', ha='right', va='top')
pyplot.ylabel('y', va='top', rotation='horizontal')
pyplot.gca().set_aspect('equal')

pyplot.subplot(212)
pyplot.scatter(x, y, c=labels)
pyplot.xlabel('x', ha='right', va='top')
pyplot.ylabel('y', va='top', rotation='horizontal')
pyplot.gca().set_aspect('equal')

pyplot.show()
