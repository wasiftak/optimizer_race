# path: app.py
import streamlit as st

from ui import landscape_tab, model_tab

st.set_page_config(page_title='Optimizer Racing League', layout='wide')
st.title('Optimizer Racing League')
st.caption('UCS668 project: build racers, watch optimizers compete')

# A radio switch instead of st.tabs: tabs would draw both Garages in the sidebar at once.
track = st.radio('Track', ['Landscape Track', 'Real Model Track'], horizontal=True,
                 label_visibility='collapsed', key='track')

if track == 'Landscape Track':
    landscape_tab.render()
else:
    model_tab.render()