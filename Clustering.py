# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 22:29:42 2017

@author: Tejal
"""
# This program can be used to compare the performance of the 3 clustering algorithms
# considered while building the final recommendation system. 
# 1 - HAC Agglomerative clustering analysis with average linkage
# 2 - Affinity propogation analysis 
# 3 - Spectral clustering analysis
# It uses the similarity matrix constructed using Pearson correlation as distance for clustering
# The output shows the MAE and coverage for the chosen clustering algorithm. 
# Cluster size = 25 is chosen for HAC and spectral clustering

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
import pickle
import math


def predict():


    test_ratings= np.zeros([len(test_data),2])
    not_rated = []
    for j,rec in enumerate(test_data):
        if j%10000==0:
            print "In movie %d"%j
    #   Get the movie and user sequential number to query the user-item and movie-similarity tables 
        user = users[rec[0]]
        movie = movies[rec[1]]
        clus = clus_labels[movie]
        clus_movie = np.where(clus_labels==clus)
        rated_movies = np.where(user_item[user]!=0)
        rated_movies = list(set(clus_movie[0]).intersection(set(rated_movies[0])))
        
#       If user has not rated any movies in cluster, assign the average of all movie ratings
#       given by user as the predicted rating for particular movie 
        if (len(rated_movies)==0):
            not_rated.append(rec[1])
            rated_movies = np.where(user_item[user]!=0)
        ratings = user_item[user,rated_movies]
        sim = sim_mat[movie,rated_movies]        
    #  Create matrix of movies rated by user and their similarity with the movie to be rated
        rated = np.column_stack((ratings,sim))
        pred_rating = np.dot(rated[:,0],rated[:,1])*1.0/sum(rated[:,1])
        pred_rating = round(pred_rating * 2) / 2 
        if(math.isnan(pred_rating)   ):
            pred_rating = 0
        test_ratings[j]=[pred_rating,rec[2]]
        
    print("Prediction complete")

# Mean Absolute Error Computation
    mae = sum(abs(test_ratings[:,0]-test_ratings[:,1]))*1.0/len(test_ratings)
    return (mae,not_rated)

def base_predict():
    not_rated = []
    test_ratings= np.zeros([len(test_data),2])
    for j,rec in enumerate(test_data):
    #   Get the movie and user sequential number to query the user-item and movie-similarity tables 
        user = users[rec[0]]
        movie = movies[rec[1]]
        rated_movies = np.where(user_item[user]!=0)
        if (len(rated_movies)==0):
            not_rated.append(rec[1])
            pred_rating = 0
            test_ratings[j]=[pred_rating,rec[2]]
        else:
            ratings = user_item[user,rated_movies[0]]
            sim = sim_mat[movie,rated_movies[0]]
        #  Create matrix of movies rated by user and their similarity with the movie to be rated
            rated = np.column_stack((ratings,sim))
            pred_rating = np.dot(rated[:,0],rated[:,1])*1.0/sum(rated[:,1])
            pred_rating = round(pred_rating * 2) / 2
                
            test_ratings[j]=[pred_rating,rec[2]]
    print("Prediction complete")
    
    # Mean Absolute Error Computation
    mae = sum(abs(test_ratings[:,0]-test_ratings[:,1]))*1.0/len(test_ratings)
    return(mae,not_rated)
    
    
    
################ Begin of main #############
prompt = "Enter\n 1 - for agglomerative clustering analysis\n 2 - for affinity propogation analysis \n 3 -for Spectral clustering analysis \n Input - "
inp = raw_input(prompt)   
print("Computing values...")
data2 = []
with open("Pickle_file1", "rb") as f:
    for _ in range(pickle.load(f)):
        data2.append(pickle.load(f))

sim_mat = np.genfromtxt("sim_mat_Pearson.csv",delimiter=",")        
train_data = data2[0]
test_data = data2[1]
users = data2[2]
movies = data2[3]
user_item = data2[4]

if inp == '1': 
    cl = AgglomerativeClustering(affinity="precomputed",linkage='average', n_clusters=25).fit(sim_mat)
    clus_labels = cl.labels_

elif inp =='2':
    af = AffinityPropagation(verbose=True,affinity="precomputed").fit(sim_mat)             
    clus_labels = af.labels_
else:
    sc = SpectralClustering(25,affinity="precomputed").fit(sim_mat)
    clus_labels = sc.labels_

mae,not_rated = predict()
print("MAE = %f"%mae)
cov = (len(test_data)-len(not_rated))*1.0/len(test_data)
print("coverage = %f"%cov)
