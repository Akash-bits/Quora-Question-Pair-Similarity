import streamlit as st
import pickle
import helper

pickle_off = open("model.pkl","rb")
rf = pickle.load(pickle_off)

st.header('Quora Question Pair Similarity')

q1 = st.text_input('Enter 1st question')
q2 = st.text_input('Enter 2nd question')

if st.button('Predict Similarity'):
    query = helper.query_point_creator(q1,q2)
    result = model.predict(query)[0]

    if result:
        st.header('Similar Questions')
    else:
        st.header('Not Similar Questions')
