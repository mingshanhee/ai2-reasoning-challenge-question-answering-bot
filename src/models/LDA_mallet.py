import pandas as pd
import numpy as np
import gensim
import nltk
import ast
import os
import datetime
from gensim.models import CoherenceModel
import pickle

def docs2vecs(docs, dictionary):
    # docs is a list of documents returned by corpus2docs.
    # dictionary is a gensim.corpora.Dictionary object.
    vecs1 = [dictionary.doc2bow(doc) for doc in docs]
    return vecs1

def docs2vecs(docs, dictionary):
    # docs is a list of documents returned by corpus2docs.
    # dictionary is a gensim.corpora.Dictionary object.
    vecs1 = [dictionary.doc2bow(doc) for doc in docs]
    return vecs1

def get_docs_topics_dist(all_topics):
  #get a doc:topic dictionary
  document_topic_relation = {}
  i = 0
  for doc_topics, word_topics, phi_values in all_topics:
      document_topic_relation[i] = doc_topics
      i += 1

  return document_topic_relation

def get_topics_docs_dist(document_topic_relation):
  #get a topic:docs dictionary from doc:topics dictionary
  topics_docs = {}

  for i in range(0,100):
   topics_docs[i] = []

  for k, v in document_topic_relation.items():
    for i in range(0,len(v)):
      if topics_docs[v[i][0]] != []:
        topics_docs[v[i][0]].append([k, v[i][1]])
      else:
        topics_docs[v[i][0]] = [[k, v[i][1]]]

  return topics_docs


def sort_based_on_percent(sub_li): 
  #sorting
  sub_li.sort(key = lambda x: x[1], reverse = True) 
  return sub_li 
  
#snippet below for running on colab
# def install_java():
#   !apt-get install -y openjdk-8-jdk-headless -qq > /dev/null      #install openjdk
#   os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"     #set environment variable
#   !java -version       #check java version
# install_java()

# !wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
# !unzip mallet-2.0.8.zip  

# because we mounted at gdrive/My Drive, need to move mallet-2.0.8 from there to content folder

# os.environ['MALLET_HOME'] = '/content/mallet-2.0.8'
# mallet_path = '/content/mallet-2.0.8/bin/mallet' 

#snippet for desktop
# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip\
os.environ.update({'MALLET_HOME':r'C:/Users/Shawn/Desktop/mallet-2.0.8'}) #change to location where mallet is
mallet_path = r'C:\\Users\\Shawn\\Desktop\\mallet-2.0.8\\bin\\mallet' # update this path

df_csl = pd.read_csv(' ') #include path to cleaned file

NA_lemmatized_list = [ast.literal_eval(i) for i in df_csl['Lemmatized']]

dict_NA_lemmatized = gensim.corpora.Dictionary(NA_lemmatized_list) #generating dictionary
vecs_NA_lemmatized = docs2vecs(NA_lemmatized_list, dict_NA_lemmatized) #generating vectors

#for timing duration of model, can remove if unnecessary
start = datetime.datetime.now()
print(start)

ldamallet_50 = gensim.models.wrappers.LdaMallet(mallet_path, corpus=vecs_NA_lemmatized, num_topics=50, id2word=dict_NA_lemmatized)

#saving model to pickle, can load next time if required, no need run model again
pickle.dump(ldamallet_50,open("TM/Project/data/ldamallet_50_lemma.pickle",'wb')) #update file path

timetaken = datetime.datetime.now() - start
print("Time taken for modelling: ", timetaken)

start = datetime.datetime.now()
print(start)

coherence_model_lda_mallet50 = CoherenceModel(model=ldamallet_50, texts=NA_lemmatized_list, dictionary=dict_NA_lemmatized, coherence='c_v')
coherence_lda_mallet50 =coherence_model_lda_mallet50.get_coherence()

print("Coherence")
print('50: ' , coherence_lda_mallet50)
timetaken = datetime.datetime.now() - start
print("Time taken for coherence: ", timetaken)

#show topics if required
# topics = ldamallet_50.show_topics(50, 15)
# for i in range(0, 50):
#     print(topics[i])

# #converting to lda model from ldamallet to make use of lda methods
# ldamallet_to_lda_100 = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet_100, gamma_threshold=0.001, iterations=50)

# #get list of doc_topics, word_topics, phi_values
# all_topics = ldamallet_to_lda_100.get_document_topics(vecs_NA_lemmatized,minimum_probability=0.01, per_word_topics=True)

# #get topic_document distribution  
# document_topic_relation = get_docs_topics_dist(all_topics)
# topics_docs = get_topics_docs_dist(document_topic_relation)

# #sort based on highest score for each topic
# for k,v in topics_docs.items():
#   v =  sort_based_on_percent(v)

# final 
# # topics_docs