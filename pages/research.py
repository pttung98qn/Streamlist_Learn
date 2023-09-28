import streamlit as st
import requests, json
import concurrent.futures
import pandas as pd

page_title = 'Mini tool Keyword research'
st.set_page_config(
    page_title=page_title,
    page_icon="https://seoreporter.site/wp-content/uploads/2023/06/cropped-favicon-32x32.jpg",
)

if 'current_page' not in st.session_state or  page_title != st.session_state['current_page']:
	st.session_state['current_page'] = page_title
	st.session_state['first_run'] = True


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
		if len(full_results)>=max_result:
			break
		new_search = list(new_search)[:round((max_result-len(full_results))/3)]
	return full_results

st.header(":rainbow[Tìm kiếm longtail keyword]", divider="rainbow")
cols = st.columns([2,10])
lang = cols[0].selectbox(":blue[Chọn Ngôn ngữ]", ('vi','en'))
input_key = cols[1].text_input(label=":blue[nhập từ khóa]", placeholder="search...")
st.divider()
result_count = st.text("")
result_output = st.text("")

if input_key or lang:
	if input_key:
		with st.spinner('Loading...'):
			data = list(search(lang, input_key, max_deep = MAX_DEEP, max_result = MAX_RESULT))
			df = pd.DataFrame(data=data, columns=["Keyword"])

			data_len = len(data)
			result_count.subheader(f"Tìm thấy {data_len}+ từ khoá")
			result_output.dataframe(df, use_container_width=True)

