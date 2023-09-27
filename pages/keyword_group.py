import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np
import csv, os
import pandas as pd

model_name = 'bert-base-multilingual-uncased'
model_path = './bert_model'
if not os.path.exists(model_path):
	os.makedirs(model_path, exist_ok=True)
	model = SentenceTransformer(model_name)
	model.save(model_path)
else:
	model = SentenceTransformer(model_path)
	
TRY_NUM = 30
DATA_NUM = 500

def keyword_grouping(list_key):
	data_len = len(list_key)

	max_step = round(data_len/3)
	step = round(max_step/TRY_NUM)
	step = step if step > 1 else 1
	k_values = range(2, max_step, step)

	best_score = -1
	best_k = None
	g_model = None

	embeddings = model.encode(list_key)
	for k in k_values:
		kmeans = MiniBatchKMeans(n_clusters=k, random_state=0)
		kmeans.fit(embeddings)
		labels = kmeans.labels_
		score = silhouette_score(embeddings, labels)
		if score > best_score:
			best_score = score
			best_k = k
			g_model = kmeans

	labels = g_model.labels_

	output = []
	for i in range(len(labels)):
		output.append([list_key[i], int(labels[i])])
	output = sorted(output, key=lambda x: x[1])

	return output


st.header("Nhóm từ khóa bằng ngữ nghĩa", divider="rainbow")
input_data = st.text_area(label="Nhập danh sách từ khóa", height=300)
st.divider()
st.subheader(":blue[Kết quả nhóm key]")
if input_data:
	input_data = input_data.split('\n')
	result_output = st.text("loading...")
	result = keyword_grouping(input_data)
	df = pd.DataFrame(data=result, columns=['keywords', 'group'])
	result_output.dataframe(df, use_container_width=True)

