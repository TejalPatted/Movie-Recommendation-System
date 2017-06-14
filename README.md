# Movie-Recommendation-System
This is a hybrid Movie Recommendation System, implemented using Memory based Collaborative Filtering and Model based Collaborative Filtering. 
For the memory based approach, item-based filtering technique was implemented using item-item similarity matrix. Cosine similarity, Pearson Correlation similarity and Adjusted Cosine Similarity algorithms were compared based on MAE. The final model uses Pearson Correlation similarity for item based similarity computation.
The movies were then clustered to enhance the response time of recommendation. Hierarchical Agglomerative Clustering, Spectral Clustering and Affinity Propagation Clustering techniques were compared based on MAE and coverage. In the final model Affinity Propagation clustering was used due to the benefit of implicitly choosing the cluster size.

File description:
1. Clustering.py - It compares the 3 clustering algorithms
2. Data_chunk.py - It is the Data wrangling code which selects users who have rated atleast 100 movies.
3. Recommendation.py - Final recommendation model
4. similarity.py - It compares the similarity algorithms

Data and relevant file for the project can be accessed at - https://drive.google.com/open?id=0B-QvfxcJ18f_Sk4wazI4Q1hJLTg
