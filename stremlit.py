import streamlit as st

pages = [
        st.Page("pag1.py", title="Demand"),
        st.Page("pag2.py", title="inventry"),
    ]

pg = st.navigation(pages)
st.sidebar.image("D:/Book_cpny/mysql/book.jpg")
pg.run()


        




