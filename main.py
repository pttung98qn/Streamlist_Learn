import streamlit as st
# import views__research, views__keyword_group

st.set_page_config(
    page_title="Mini keyword tool",
    page_icon="https://seoreporter.site/wp-content/uploads/2023/06/cropped-favicon-32x32.jpg",
)
st.header(body=":rainbow[Chào mừng đến với Mini keyword tool]", divider='rainbow')
st.write(':blue[Đây là phần mềm thử nghiệm các tính năng trên SEO Reporter]')
cols = st.columns(2)

cols[0].link_button("Keyword Research",'/research', use_container_width =True, type="secondary")
cols[1].link_button("Gom nhóm từ khóa",'/keyword_group', use_container_width =True, type="secondary")