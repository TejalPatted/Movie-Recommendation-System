# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 20:35:53 2017

@author: Tejal
"""

# This program recommends to a user top 10 movies, similar to the entered movie
# There are 2 options to execute this program -
# 1. You can compute similarity matrix and movie clusters using Pearson similarity algorithm
#    and Agglomerative clustering and use the components to get recommendations
# 2. You can use the pre-computed similarity matrix and clusters that were generated using
#    same program and have been saved in the project folder
# Both methods yield same results, only execution time varies

import numpy as np
import pickle
from sklearn.cluster import AffinityPropagation
import math
import sys

train_data=[]
test_data = []
users ={}
movies={}

# Get Movielens filtered data to compute similarity
def get_data():
    train_data = np.genfromtxt("final_movie_data.csv",delimiter= ',',skip_header=(1))
    
# Assign sequential number to users and movies and store it as key value pair
# Sequential numbers are needed so that matrix can be created as user_num X movie_num
 
    user_id = list(set(train_data[:,0]))
    user_id.sort()
    movie_id = list(set(train_data[:,1]))
    movie_id.sort()
    users={}
    movies={}
    
# movies_id:sequential movie number
    for i,j in enumerate(user_id):
        users[j]=i
    for i,j in enumerate(movie_id):
    
#        print(i)
        movies[j]=i
    
    # Create user item matrix
    user_item = np.empty((len(set(train_data[:,0])),len(set(train_data[:,1]))))
     
    for row in train_data:
        i = users[int(row[0])]
        j = movies[int(row[1])]              
        user_item[i][j]=row[2]
    return(train_data,users,movies,user_item)
  
# Compute similarity matrix    
def find_similarity():  
    movie_sim = np.zeros([len(movies.keys()),len(movies.keys())])
    for st,m1 in enumerate(movies.keys()):
        if st%1000==0:
            print 'in movie',st        
            
#       r1 - Average rating of movie m1
        r1 = np.average(user_item[:,movies[m1]])
        u_m1 = np.where(user_item[:,movies[m1]]!=0)

# Find co-ratings of movie 1 and movie 2            
# Fetch users who rated movie 1 and movie 2 separately and take a intersection            
        for j in range(st,len(movies.keys())):
            
            m2 = movies.keys()[j]
            r2 = np.average(user_item[:,movies[m2]])            
            u_m2 = np.where(user_item[:,movies[m2]]!=0)
            u = list(set(u_m1[0]).intersection(set(u_m2[0])))            

# Pearson similarity
            if len(u)!=0:
                co_ratings = user_item[np.ix_(u,[int(movies[m1]),int(movies[m2])])]
                num = sum((co_ratings[:,0]-r1)*(co_ratings[:,1]-r2))
                den = ((sum((co_ratings[:,0]-r1)**2))**0.5)*((sum((co_ratings[:,1]-r2)**2))**0.5)
                corr = num*1.0/den
                movie_sim[st][j] = corr
                if j != st:
                    movie_sim[j][st] = corr
             
            
    return(movie_sim)  
    
# Computing top-10 recommendation for the user    
def compute_reco(act_user,act_mov):
    
    user = users[act_user]
    movie = movies[act_mov]
    clus = clus_labels[movie]
    clus_movie = np.where(clus_labels==clus)
    user_rated_movies = np.where(user_item[user]!=0)
    rated_movies = list(set(clus_movie[0]).intersection(set(user_rated_movies[0]))) 
    if movie in rated_movies:
        rated_movies.remove(movie)
    clus_movie = np.delete(clus_movie,np.where(clus_movie[0]==movie))
    ratings = user_item[user,rated_movies]
#    dtype = [('movie_num', int), ('rating', float), ('W', int)]
    reco_ratings = np.zeros([len(clus_movie),3])
    for j,m in enumerate(clus_movie):
        if m in user_rated_movies[0]:
            pred_rating = user_item[user,m]
            reco_ratings[j]=[0,m,pred_rating]

        else:        
            sim = sim_mat[m,rated_movies]
            rated = np.column_stack((ratings,sim))
            pred_rating = np.dot(rated[:,0],rated[:,1])*1.0/sum(rated[:,1])
            pred_rating = round(pred_rating * 2) / 2         
            if(math.isnan(pred_rating)):
                pred_rating = 0            
            reco_ratings[j]=[1,m,pred_rating]
    not_watched_ind = np.where(reco_ratings[:,0]==1)
    if(len(not_watched_ind[0])>10):
        not_watched = reco_ratings[not_watched_ind[0]]
        reco_movies = not_watched[np.argsort(not_watched[:, 2])][::-1]
        reco_movies= reco_movies[0:10]
    elif (len(reco_ratings)>10):
        reco_movies = reco_ratings[np.argsort(reco_ratings[:, 2])][::-1]
        reco_movies= reco_movies[0:10]
    else:
        reco_movies = reco_ratings[np.argsort(reco_ratings[:, 2])][::-1]
    final_list = [0]*len(reco_movies)
    for k,i in enumerate(reco_movies):
        final_list[k] = next(key for key, value in movies.iteritems() if value == i[1] )
    return(final_list)
    
   
    
#################### Begin of program #######################     
prompt = "Do you want to compute similarity matrix and cluster?\n Enter Y - To compute the components \n Enter N - To use precomputed components\n Enter ex to stop execution \nInput - "
while True:
    inp_choice = raw_input(prompt)
    if inp_choice.lower() == 'y':
#   Read the data
        train_data,users,movies,user_item = get_data()
        print("Data read complete\n Computing similarity...")
        
#   Compute similarity matrix
        sim_mat = find_similarity()   
        print("Similarity matrix computed\n Movies being clustered...")
        np.savetxt("sim_mat_Pearson.csv",sim_mat,delimiter=',')        

#   Cluster the movies     
        af = AffinityPropagation(verbose=True,affinity="precomputed").fit(sim_mat)             
        clus_labels = af.labels_   
        print("Movies clustered")
       
        break
    elif inp_choice.lower()=='n':
        print("Loading precomputed components...")
        data2 = []
        with open("Pickle_file", "rb") as f:
            for _ in range(pickle.load(f)):
                data2.append(pickle.load(f))    
        sim_mat = np.genfromtxt("sim_mat_Pearson.csv",delimiter=",")        
        train_data = data2[0]
#        test_data = data2[1]
        users = data2[1]
        movies = data2[2]
        user_item = data2[3]
        clus_labels = data2[4]
        print("\nPrecomputed components loaded")
        break
    elif inp_choice.lower()=="ex":
        sys.exit("Program stopped as requested")
    else:
        print("Invalid input")
        continue
    
while True:
    print "\n1st 100 User ids =", users.keys()[0:100],
    print"\n"
    print "1st 100 Movie ids =",movies.keys()[0:100],
#             
    inp = raw_input("\nEnter user id and movie id separated by comma-   ")   
    if inp == "":      
        break           
    else:
        act_user,act_mov = inp.split(',')
        act_user = int(act_user)
        act_mov = int(act_mov)
        final_list = compute_reco(act_user,act_mov)
        print("Top %d movies for user %d similar to movie %d \n"%(len(final_list),act_user,act_mov))
        print(final_list)
        inp1 = raw_input("Do you want to continue? Y/N - ")
        if inp1.lower()== 'y':
            continue
        else:
            break
            