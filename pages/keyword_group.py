import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np
import csv, os
import pandas as pd
import time

page_title = 'Mini tool Keyword group'
st.set_page_config(
    page_title= page_title ,
    page_icon="https://seoreporter.site/wp-content/uploads/2023/06/cropped-favicon-32x32.jpg",
)
if 'current_page' not in st.session_state or  page_title != st.session_state['current_page']:
	st.session_state['current_page'] = page_title
	st.session_state['first_run'] = True



print('step 0')
# @st.cache_resource
# def get_model(model_name, model_path):
# 	print(f'start get model {model_name}')
# 	if not os.path.exists(model_path):
# 		print('download model')
# 		os.makedirs(model_path, exist_ok=True)
# 		model = SentenceTransformer(model_name)
# 		model.save(model_path)
# 	else:
# 		print('load model')
# 		model = SentenceTransformer(model_path)
# 	print(f'done get model {model_name}')
# 	return model
# print('step 1')

@st.cache_resource
def get_model(model_name, model_path):
	print(f'start get model {model_name}')
	os.makedirs(model_path, exist_ok=True)
	model = SentenceTransformer(model_name)
	print(f'done get model {model_name}')
	return model

vi_model = get_model('vinai/phobert-large', './bert_model/phobert_large')
base_model = get_model('bert-base-multilingual-uncased', './bert_model/bert_base_multilingual')
# en_model = get_model('bert-base-uncased', './bert_model/bert_base_uncased')

DATA_NUM = 500
MAX_TEST = 15
print('step 2')

def get_range_loop(k_values):
	output = [item for item in k_values]
	if k_values.stop not in output:
		output.append(k_values.stop)
	return output


def get_k_values(min, max):
	step = round((max-min)/5)
	step = step if step>1 else 1
	return range(min, max, step)

def get_new_k_values(k_values, best_k):
	k_values = get_range_loop(k_values)
	i = k_values.index(best_k)
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
	return get_k_values(min, max)

def grouping(embeddings, k_values):
	best_score = -1
	best_k = -1
	loop_range = get_range_loop(k_values)
	for k in loop_range:
		if k in st.session_state['kmean_list']:
			kmeans = st.session_state['kmean_list'][k]
		else:
			kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, n_init="auto")
			kmeans.fit(embeddings)
			st.session_state['kmean_list'][k] = kmeans

		labels = kmeans.labels_
		score = silhouette_score(embeddings, labels)
		st.session_state['silhouette_score'][k] = score
		if score > best_score:
			best_score = score
			best_k = k
	
	return best_k

def keyword_grouping():
	print("______start keyword_grouping")
	list_key = st.session_state['input_data']
	data_len = len(list_key)

	start_time = time.time()
	if lang=='order/multiple language':
		embeddings = base_model.encode(list_key)
	elif lang=='vi':
		embeddings = vi_model.encode(list_key)
	# elif lang == 'en':
	# 	embeddings = en_model.encode(list_key)
	st.session_state['embeddings'] = embeddings

	embeddings_time = time.time()
	st.session_state['time_run']['embeddings'] = embeddings_time-start_time

	progress_e.progress(30)

	k_values = get_k_values(5, data_len)
	while True:
		best_k = grouping(embeddings, k_values)
		if k_values.step==1:
			break
		k_values = get_new_k_values(k_values, best_k)

	group_time = time.time()
	st.session_state['time_run']['group_time'] = group_time-embeddings_time
	return best_k

def get_k_cluster_data(k):
	list_key = st.session_state['input_data']
	g_model = st.session_state['kmean_list'][k]
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

def load_out_data(k):
	result = get_k_cluster_data(k)
	parent_count = len(set([item[1] for item in result]))	
	parent_count_e.header(f":orange[{parent_count}]")
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
	ss_data = st.session_state['silhouette_score']
	ss_data_list = [[int(item),ss_data[item]] for item in ss_data]
	ss_data_list = sorted(ss_data_list, key=lambda x:x[0])
	ss_data_df = pd.DataFrame(ss_data_list, columns=['k', 'score'])

	chart_e[0].line_chart(data=ss_data_df, x='k', y='score', use_container_width=True, height=400)
	chart_e[1].dataframe(ss_data_df, use_container_width=True, height=350, hide_index=True)

	time_run = st.session_state['time_run']
	embeddings_time = round(time_run['embeddings'],3)
	group_time = round(time_run['group_time'],3)
	time_run_e[0].text(f'embeddings time : {embeddings_time}')
	time_run_e[1].text(f'group time  : {group_time}')
	time_run_e[2].selectbox("Chọn k: ", tuple([item[0] for item in ss_data_list]), key="select_k")
	
def run():
	if st.session_state['reload']:
		st.session_state['silhouette_score'] = dict(())
		st.session_state['kmean_list'] = dict(())
		st.session_state['time_run'] = {'embeddings':0, 'group_time':0}

	input_data = st.session_state['input_data']
	input_data_length = len(input_data)
	summary_e[0].header(f":orange[{input_data_length}]")
	if input_data_length>500 or input_data_length<15:
		warning_e.warning(f"Lỗi: Số lượng từ khóa phải lớn hơn 15 và nhỏ hơn 500, số lượng bạn nhập là: {input_data_length}")
	else:
		with st.spinner('Loading...'):
			parent_count_e.header(":orange[...]")
			progress_e.progress(5)
			if st.session_state['reload']:
				best_k = keyword_grouping()
			else:
				best_k = st.session_state['select_k']
			progress_e.text('')
			load_out_data(best_k)

st.session_state['first_run'] = False
st.header(":rainbow[Nhóm từ khóa qua ngữ nghĩa]", divider="rainbow")
st.warning("Nếu từ khóa chỉ bao gồm tiếng việt thì chọn 'vi', nếu là tiếng anh, ngôn ngữ khác hoặc kết hợp nhiều ngôn ngữ thì chọn 'order/multiple language'")
lang = st.selectbox(":blue[Ngôn ngữ]", ("vi", "order/multiple language"))
input_data = st.text_area(label=":blue[Nhập danh sách từ khóa]", height=250, placeholder="Tỗi dòng là 1 từ khóa")
st.divider()
warning_e = st.text("")
st.subheader(":blue[Kết quả nhóm key]")
progress_e = st.text("")
summary_e = st.columns(3)
summary_e[0].markdown("Tổng từ khóa")
summary_e[1].markdown("Tổng parent")
parent_count_e = summary_e[1].header("")
st.divider()
chart_e = st.columns([9,3])
time_run_e = st.columns(3)
st.warning("'k' là số nhóm, mặc định k được chọn sẽ có score lớn nhất, Bạn có thể test các 'k' khác nhau để thử nghiệm. Khuyến khích chọn các 'k' là đỉnh sóng trong biểu đồ.")

group_e, list_e  = st.tabs(['Group', 'List'])


if input_data:
	input_data = input_data.split('\n')
	input_data = list(set([item for item in input_data if item!='']))
	if 'input_data' not in st.session_state or 'lang' not in st.session_state or 'select_k' not in st.session_state:
		st.session_state['reload'] = True
		st.session_state['input_data'] = input_data
		st.session_state['lang'] = lang
	elif input_data == st.session_state['input_data'] and lang==st.session_state['lang']:
		st.session_state['reload'] = False
	else:
		st.session_state['reload'] = True
		st.session_state['input_data'] = input_data
		st.session_state['lang'] = lang
	run()
