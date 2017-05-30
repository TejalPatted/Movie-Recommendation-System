# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 20:39:40 2017

@author: Tejal
"""

# This program is to evaluate various similarity compuation algorithms. 3 algorithms are used
# 1 - Pearson correlation
# 2 - Cosine similarity
# 3 - Adjusted cosine similarity 
# To evaluate the performance of the similarity metrics, the built similarity matrix 
# is used to predict ratings of user-movie pair in test dataset using weighted sum prediction
# Mean square errors is used to assess the performance

import numpy as np

train_data=[]
test_data = []
users ={}
movies={}

def get_data():
    data = np.genfromtxt("final_movie_data.csv",delimiter= ',',skip_header=(1))
    
    # Split data into train and test data in 80:20 ratio
    np.random.shuffle(data)
    ind = int(.8*len(data))
    train_data=data[0:ind]
    test_data = data[ind+1:len(data)]
    
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
    return(train_data,test_data,users,movies,user_item)   

def find_similarity_cos():
    movie_sim = np.zeros([len(movies.keys()),len(movies.keys())])
    for st,m1 in enumerate(movies.keys()):
        if st%1000 ==0:
            print 'in movie',st
        u_m1 = np.where(user_item[:,movies[m1]]!=0)
#        for j,m2 in enumerate(self.movies.keys()[st:len(self.movies.keys())]):
# Find co-ratings of movie 1 and movie 2            
# Fetch users who rated movie 1 and movie 2 separately and take a intersection            
        for j in range(st,len(movies.keys())):
            m2 = movies.keys()[j]
            u_m2 = np.where(user_item[:,movies[m2]]!=0)
            u = list(set(u_m1[0]).intersection(set(u_m2[0])))            

# Cosine based similarity
            if len(u)!=0:
                co_ratings = user_item[np.ix_(u,[int(movies[m1]),int(movies[m2])])] 
                prod_m1_m2 = np.dot(co_ratings[:,0],co_ratings[:,1].T)
                m1_len = np.linalg.norm(co_ratings[:,0])
                m2_len = np.linalg.norm(co_ratings[:,1])
                sim_m1_m2 = prod_m1_m2/(m1_len*m2_len)
                movie_sim[st][j] = sim_m1_m2
                if j != st:
                    movie_sim[j][st] = sim_m1_m2 
    return(movie_sim)

def find_similarity_adjcos():  
    movie_sim = np.zeros([len(movies.keys()),len(movies.keys())])
    user_avg_rating = np.zeros([user_item.shape[0],1])
    
    for u,j in enumerate(user_item[:,0]):
        user_avg_rating[u] = np.average(user_item[u,:])       
       
    for st,m1 in enumerate(movies.keys()):
        if st%1000 ==0:
            print 'in movie',st
            
        u_m1 = np.where(user_item[:,movies[m1]]!=0)

# Find co-ratings of movie 1 and movie 2            
# Fetch users who rated movie 1 and movie 2 separately and take a intersection            
        for j in range(st,len(movies.keys())):
            
            m2 = movies.keys()[j]
         
            u_m2 = np.where(user_item[:,movies[m2]]!=0)
            u = list(set(u_m1[0]).intersection(set(u_m2[0])))            

# Cosine based similarity
            if len(u)!=0:
                co_ratings = user_item[np.ix_(u,[int(movies[m1]),int(movies[m2])])]
                # User average ratings of users who have co-rated movie m1 and m2
                u_a_r = user_avg_rating[u]
                num = np.dot((co_ratings[:,0]-u_a_r[:,0]),(co_ratings[:,1]-u_a_r[:,0]).T)
                den = ((sum((co_ratings[:,0]-u_a_r[:,0])**2))**0.5)*((sum((co_ratings[:,1]-u_a_r[:,0])**2))**0.5)
                corr = num*1.0/den
                movie_sim[st][j] = corr
                if j != st:
                    movie_sim[j][st] = corr             
            
    return(movie_sim)

              
                
def find_similarity_pearson():  
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

# Predict ratings for test data using weighted sum of similarity ratings
# Calculate MAE to assess the performance    
def predict():

    test_ratings= np.zeros([len(test_data),2])
    for j,rec in enumerate(test_data):
    #   Get the movie and user sequential number to query the user-item and movie-similarity tables 
        user = users[rec[0]]
        movie = movies[rec[1]]
        rated_movies = np.where(user_item[user]!=0)
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
    return (mae)
        
############# Begin of main ###############

prompt = "Enter\n 1 - Similarity using cosine algorithm\n 2 - Similarity using normalized cosine algorithm\n 3 - Similarity using Pearson correlation\n Input - "
inp = raw_input(prompt)    
    
    
train_data,test_data,users,movies,user_item = get_data()
#
print('Data loaded')

if inp =='1':
    sim_mat = find_similarity_cos()
elif inp == '2':
    sim_mat = find_similarity_adjcos()
else:
    sim_mat = find_similarity_pearson()

print("Similarity computation complete")

#sim_mat = find_similarity_cos()
#mae = predict()
#print("Error of cos = %f"%mae)  
#sim_mat = find_similarity_adjcos()
#mae = predict()
#print("Error of adj cos = %f"%mae)
#sim_mat = find_similarity_pearson()

mae = predict()
print("MAE = %f"%mae)