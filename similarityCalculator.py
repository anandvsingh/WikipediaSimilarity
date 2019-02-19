import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
import pickle
import sys

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
with tf.Session() as session:
	session.run([tf.global_variables_initializer(), tf.tables_initializer()])
	message_embeddings = session.run(embed(messages)) # In message embeddings each vector is a second (1,512 vector) and is numpy.ndarray (noOfElemnts, 512)
