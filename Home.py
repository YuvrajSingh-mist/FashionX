import streamlit as st
import time
import pandas  as pd
import pickle
import nltk
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from streamlit_card import card

st.set_page_config(
    page_title = 'FashionX'
)


st.markdown("<h1 style='text-align: center; color: white;'>FashionX</h1>", unsafe_allow_html=True)
st.divider()


