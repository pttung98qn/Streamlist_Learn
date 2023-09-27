import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np
import csv, os
import pandas as pd

st.set_page_config(
    page_title="Mini tool Keyword group",
    page_icon="https://seoreporter.site/wp-content/uploads/2023/06/cropped-favicon-32x32.jpg",
)


model_name = 'bert-base-multilingual-uncased'
model_path = './bert_model'
@st.cache_resource
def get_model():
	print('start get model')
	if not os.path.exists(model_path):
		print('download model')
		os.makedirs(model_path, exist_ok=True)
		model = SentenceTransformer(model_name)
		model.save(model_path)
	else:
		print('load model')
		model = SentenceTransformer(model_path)
	print('done get model')
	return model
model = get_model()
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
	progress_e.progress(30)

	i = 0
	for k in k_values:
		progress_e.progress(round(30+i*70/30))
		kmeans = MiniBatchKMeans(n_clusters=k, random_state=0)
		kmeans.fit(embeddings)
		labels = kmeans.labels_
		score = silhouette_score(embeddings, labels)
		if score > best_score:
			best_score = score
			best_k = k
			g_model = kmeans
		i+=1
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
input_data = st.text_area(label=":blue[Nhập danh sách từ khóa]", height=250, placeholder="Tỗi dòng là 1 từ khóa")
st.divider()
warning_e = st.text("")
result_header =st.text("")
progress_e = st.text("")
summary_e = st.columns(3)
show_mode_e = st.text("")

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

			list_e, group_e = show_mode_e.tabs(['List', 'Group'])

			
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

