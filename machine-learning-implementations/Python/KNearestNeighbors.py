# -*- coding: utf-8 -*-
"""
K Nearest Neighbors

Created on Wed Sep  9 19:04:43 2020

@author: Joás de Brito Ferreira Filho
"""

from statistics import mode


## MANHATTAN DISTANCE (L1 Norm)
def manhattan_distance(p1, p2):
    distance = 0
    for i in range(len(p1)):
        distance += abs(p1[i] - p2[i])
    return distance

# Measures the Manhattan Distance from point to all points in plane 
# and returns the k points with smallest distances to point.
# if predict is set to True, returns the label predicted by the KNN
def knn(point, plane, k, predict=False):
    distances = list()
    for p in plane:
        distances.append([p, manhattan_distance(point.coordinates, p.coordinates)])
    index = distances.argsort()[:k]
    knn = list()
    for i in index:
        knn.append(plane[i])
    if(!predict):
        return knn
    else:
        labels = list()
        for p in knn:
            labels.append(p.label)
        return mode(labels)


