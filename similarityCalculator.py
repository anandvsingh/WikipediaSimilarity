import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
import pickle
import sys
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix


try:
	with open("summary.txt", "rb") as fp:   # Unpickling
		summary = pickle.load(fp)
except Exception as e:
	print ('Cannot load the summary file, Please make sure that it exists, if not run Summary Generator first', e)
	sys.exit('Read the error message')

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(module_url)

tf.logging.set_verbosity(tf.logging.ERROR)
messages = [x[1] for x in summary]
labels = [x[0] for x in summary]
with tf.Session() as session:
	session.run([tf.global_variables_initializer(), tf.tables_initializer()])
	message_embeddings = session.run(embed(messages)) # In message embeddings each vector is a second (1,512 vector) and is numpy.ndarray (noOfElemnts, 512)

X = message_embeddings
agl = AgglomerativeClustering(n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', pooling_func='deprecated')
agl.fit(X)
dist_matrix = distance_matrix(X,X)
Z = hierarchy.linkage(dist_matrix, 'complete')
dendro = hierarchy.dendrogram(Z)