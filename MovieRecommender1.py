import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
def index_from_title(title):
    return dataset[dataset.title==title]['index'].values[0]
def title_from_index(index):
    return dataset[dataset.index==index]['title'].values[0]
#Extract similarity relation
from sklearn.metrics.pairwise import cosine_similarity
dataset=pd.read_csv('C:\\Users\\aksha\\Documents\\ML documents\\movie recommender\\movie_dataset.csv',squeeze=True)
#Deside and select parameters for input
x=dataset.iloc[:,[2,5,21,23]]
features=['genres','keywords','cast','director']
#Fill Na values with empty string
for feature in features:
        dataset[feature]=dataset[feature].fillna('')
#obtain each index values as a single string
def combined_features(row):
    return (row['genres']+" "+row['cast']+" "+row['keywords']+" "+row['director']+" ")
dataset['combined_features']=dataset.apply(combined_features,axis=1)
print(dataset['combined_features'])
#Obtain relation between strings
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
count_matrix=cv.fit_transform(dataset['combined_features'])
cosine_sim=cosine_similarity(count_matrix)
#Movie selected/watched
liked_movie="Superman Returns"
list1=[]
index=index_from_title(liked_movie)
list1 = list(enumerate(cosine_sim[index]))
list2=sorted(list1,key=lambda x:x[1],reverse=True)[0:]
for i in range(1,20):
        index2=list2[i][0]
        print(dataset[dataset.index==index2]['title'].values[0])


