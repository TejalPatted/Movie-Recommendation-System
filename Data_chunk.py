# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:26:09 2017

@author: Tejal
"""
# Data chunking program

#This program chunks the 150Mb data to select around 150000 ratings or 3000 users(which ever)
#is smaller. Only those users who have rated atleast 100 movies and only movies having 
#more than 20 ratings are selected
#The 1st output of this program provides 200578 ratings for 658 users who have rated more than 100 movies
#On an average the selected users have rated 150 movies
#Then the data is filtered to select only those movies which have more 20 ratings.
#The final data downloaded as 100031 ratings, 635 users and 2190 movies
import numpy as np

total_users = 0
flag = 0
l = []
movie = []
fin_data = np.empty((0,3))
chunk_data = np.empty((0,3))

# Iterative read file ; each time 100000 records are read and processed
for i in range(1,10):
    data = np.genfromtxt("C:\\Users\\Tejal\\Documents\\Tejal\\College\\INF552\\Project\\rating.csv",delimiter=",",max_rows=i*100000,usecols=(0,1,2),skip_header= ((i-1)*100000))
    if i ==1:
      hdr = list(data[0])
      data = np.delete(data,[0],axis=0) 
# Get unique users
    s = list(set(data[:,0]))
#Find number of occurences of the user
    for user in s:
        c = np.where(data[:,0]==user)
# Select only those users who have rated atleast 100 movies
        if total_users == 3000 or len(fin_data)>200000:
            flag = 1
            break
        if len(c[0]) > 100:
            total_users +=1
            for j in c:
                fin_data = np.append(fin_data, data[j], axis=0)
    if flag == 1:
        break

# Keep ratings of only those movies that have atleast 10 ratings

# Get unique movies
movie = list(set(fin_data[:,1]))

for i in movie:
    if len(chunk_data)>200000:
        break
    c = np.where(fin_data[:,1]==i)
#    If movie has atleast 20 ratings
    if len(c[0])>20:
        for j in c:
            chunk_data = np.append(chunk_data,fin_data[j],axis = 0)    


    
    
np.savetxt("final_movie_data.csv",chunk_data,delimiter=',')


            
     
        