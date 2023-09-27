import streamlit as st
# import views__research, views__keyword_group

st.set_page_config(
    page_title="Mini keyword tool",
    page_icon="🧊",
)
st.header(body="Chào mừng đến với Mini keyword tool", divider='rainbow')
st.write(':blue[Đây là phần mềm thử nghiệm các tính năng trên SEO Reporter]')
cols = st.columns(2)

cols[0].link_button("Keyword Research",'/research', use_container_width =True)
cols[1].link_button("Gôm nhóm từ khóa",'/keyword_group', use_container_width =True)