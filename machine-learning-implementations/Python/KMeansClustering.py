# -*- coding: utf-8 -*-
"""
K Means Clustering

Created on Fri Sep 11 14:00:31 2020

@author: Joás de Brito Ferreira Filho
"""

import random

# Manhattan Distance (L1 Norm)
def manhattan_distance(p1, p2):
    d = list()
    for i in range(len(p1)):
        d.append(abs(p1[i]-p2[i]))
    return d

# Averages the vectors' coordinates to get the centroid
def get_centroid(vectors):
    centroid = list()
    for c in range(len(vectors[0])):
        coord = 0
        for v in vectors:
            coord += v[c]
        centroid.append(coord/len(vectors))
    return centroid

# Returns k clusters containing the same vectors passed as argument
def kmc(vectors, k):
    # k randomly selected vectors, to be used in the first iteration
    centroids = random.sample(vectors, k)
    
    # list to hold k clusters
    clusters = list()
    for n in range(k):
        clusters.append(list())
    
    converged = False
    # repeat until convergence
    while(!converged):
        # for each vector determine to which centroid it is closer
        for v in vectors:
            distance_to_centroid_k = list()
            for k in centroids:
                distance_to_centroid_k.append(manhattan_distance(k, v))
                
            # index of the centroid k closest to vector v
            K = distance_to_centroid_k.argsort()[:1]
            # include v to the apropriate cluster
            clusters[K].append(v)
    
        # calculate new centroids
        centroids = list()
        for cluster in clusters:
            centroids.append(get_centroid(cluster))
            