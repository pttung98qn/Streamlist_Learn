import streamlit as st
import requests, json
import concurrent.futures
import pandas as pd

LANG = ['vi', 'en']
MAX_DEEP = 2
MAX_RESULT = 500

def longtailresearch(lang, query):
	endpoint = f"http://suggestqueries.google.com/complete/search?client=chrome&q={query}&hl={lang}"
	res = requests.get(endpoint)
	data = json.loads(res.text)
	return data[1]

def search(lang, query, max_deep = MAX_DEEP, max_result = MAX_RESULT):
	searched = set(())
	new_search = set([query])
	full_results = set(())

	i = 0
	while new_search:
		this_round_new_key = []
		futures = []
		with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
			for key in new_search:
				futures.append(executor.submit(longtailresearch, lang, key))

			for future in concurrent.futures.as_completed(futures):
				this_round_new_key.extend(future.result())

		this_round_new_key = set(this_round_new_key)
		searched = searched.union(new_search)
		full_results = full_results.union(this_round_new_key)
		new_search = this_round_new_key.difference(searched)
		i=i+1
		if i> max_deep:
		  break
	return full_results

st.header("Tìm kiếm longtail keyword", divider="rainbow")
cols = st.columns([2,10])
lang = cols[0].selectbox("Chọn Ngôn ngữ", ('vi','en'))
input_key = cols[1].text_input(label="nhập từ khóa", placeholder="search...")

if input_key or lang:
	if input_key:
		result_output = st.text("Loading data...")
		data = list(search(lang, input_key, max_deep = MAX_DEEP, max_result = MAX_RESULT))
		df = pd.DataFrame(data=data, columns=["Keyword"])
		result_output.dataframe(df, use_container_width=True)

