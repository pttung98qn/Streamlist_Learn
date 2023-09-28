import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np
import csv, os
import pandas as pd
import time

st.set_page_config(
    page_title="Mini tool Keyword group",
    page_icon="https://seoreporter.site/wp-content/uploads/2023/06/cropped-favicon-32x32.jpg",
)


@st.cache_resource
def get_model(model_name, model_path):

	print(f'start get model {model_name}')
	if not os.path.exists(model_path):
		print('download model')
		os.makedirs(model_path, exist_ok=True)
		model = SentenceTransformer(model_name)
		model.save(model_path)
	else:
		print('load model')
		model = SentenceTransformer(model_path)
	print(f'done get model {model_name}')
	return model

base_model_name = 'bert-base-multilingual-uncased'
en_model_name = 'bert-base-uncased'
vi_model_name = 'vinai/phobert-large'

base_model = get_model('bert-base-multilingual-uncased', './bert_model/bert_base_multilingual')
en_model = get_model('bert-base-uncased', './bert_model/bert_base_uncased')
vi_model = get_model('vinai/phobert-large', './bert_model/phobert_large')

DATA_NUM = 500
MAX_TEST = 15

def get_range_loop(k_values):
	output = [item for item in k_values]
	if k_values.stop not in output:
		output.append(k_values.stop)
	return output


def get_k_values(min, max):
	step = round((max-min)/5)
	step = step if step>1 else 1
	return range(min, max, step)

def get_new_k_values(k_values, i):
	k_values = get_range_loop(k_values)
	k_values_length = len(k_values)
	if i == 0:
		min = k_values[0]
		max = k_values[1]
	elif i==k_values_length-1:
		min = k_values[k_values_length-2]
		max = k_values[k_values_length-1]
	else:
		min = k_values[i-1]
		max = k_values[i+1]
	return get_k_values(min+1, max-1)

def grouping(embeddings, k_values):
	g_model = None
	best_score = -1
	best_k = -1
	loop_range = get_range_loop(k_values)
	for k in loop_range:
		kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, n_init="auto")
		kmeans.fit(embeddings)
		labels = kmeans.labels_
		score = silhouette_score(embeddings, labels)
		if score > best_score:
			best_score = score
			g_model = kmeans
			best_k = k
	return {'g_model':g_model, 'best_k_index': loop_range.index(best_k)}

def keyword_grouping(list_key):
	print("______start keyword_grouping")
	data_len = len(list_key)

	start_time = time.time()
	if lang=='global':
		embeddings = base_model.encode(list_key)
	elif lang=='vi':
		embeddings = vi_model.encode(list_key)
	elif lang == 'en':
		embeddings = en_model.encode(list_key)

	embeddings_time = time.time()

	progress_e.progress(30)

	k_values = get_k_values(2, data_len)
	g_model = None
	while True:
		output = grouping(embeddings, k_values)
		g_model = output['g_model']
		best_k_index = output['best_k_index']
		print(get_range_loop(k_values), best_k_index, get_range_loop(k_values)[best_k_index])
		if k_values.step==1:
			break
		k_values = get_new_k_values(k_values, best_k_index)

	group_time = time.time()

	labels = g_model.labels_
	output = []
	for i in range(len(labels)):
		output.append([list_key[i], int(labels[i])])
	output = sorted(output, key=lambda x: x[1])

	return output


def show_mode(mode, result, result_output):
	if mode=='list':
		df = pd.DataFrame(data=result, columns=['keywords', 'group'])
		result_output.dataframe(df, use_container_width=True)
	else:
		result_output.text(result)

st.header(":rainbow[Nhóm từ khóa qua ngữ nghĩa]", divider="rainbow")

lang = st.selectbox(":blue[Ngôn ngữ] ( :red[Chọn ngôn ngữ sẽ cho kết quả chính xác hơn] )", ("global","vi", "en"))
input_data = st.text_area(label=":blue[Nhập danh sách từ khóa]", height=250, placeholder="Tỗi dòng là 1 từ khóa")
st.divider()
warning_e = st.text("")
result_header =st.text("")
progress_e = st.text("")
summary_e = st.columns(3)
show_mode_e = st.text("")

if input_data or lang:
	if input_data:
		input_data = input_data.split('\n')
		input_data = list(set(input_data))
		input_data_length = len(input_data)
		if input_data_length>500 or input_data_length<15:
			warning_e.warning(f"Lỗi: Số lượng từ khóa phải lớn hơn 15 và nhỏ hơn 500, số lượng bạn nhập là: {input_data_length}")
		else:
			with st.spinner('Loading...'):
				result_header.subheader(":blue[Kết quả nhóm key]")
				
				summary_e[0].markdown("Tổng từ khóa")
				summary_e[0].header(f":orange[{input_data_length}]")
				summary_e[1].markdown("Tổng parent")
				total_parent = summary_e[1].header(":orange[...]")

				progress_e.progress(5)

				result = keyword_grouping(input_data)
				parent_count = len(set([item[1] for item in result]))
				total_parent.header(f":orange[{parent_count}]")

				progress_e.text('')

				group_e, list_e  = show_mode_e.tabs(['Group', 'List'])

				
				with list_e:
					df = pd.DataFrame(data=result, columns=['keywords', 'group'])
					list_e.dataframe(df, use_container_width=True)
				
				with group_e:
					result_dict = dict(())
					for line in result:
						if line[1] not in result_dict:
							result_dict[line[1]] = [line[0]]
						else:
							result_dict[line[1]].append(line[0])
					for item in result_dict:
						df = pd.DataFrame(data=result_dict[item], columns=[f'Group {item}'])
						group_e.dataframe(df, use_container_width=True, hide_index=True)

