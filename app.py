# path: app.py
import streamlit as st

st.set_page_config(page_title='Optimizer Racing League', layout='wide')
st.title('Optimizer Racing League')
st.caption('UCS668 project: build racers, watch optimizers compete')

tab_land, tab_model = st.tabs(['Landscape Track', 'Real Model Track'])

with tab_land:
    st.info('The Landscape Track arrives in Phase 1.')

with tab_model:
    st.info('The Real Model Track arrives in Phase 4.')