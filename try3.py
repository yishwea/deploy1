import streamlit as st
import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Heart Failure Prediction",page_icon="⚕️",layout="centered",initial_sidebar_state="expanded")

html_temp = """ 
    <div style ="background-color:#DDD0C8;padding:13px"> 
    <h1 style ="color:#323232;text-align:center;">Length of Stay in Hospital</h1> 
    </div> 
    """


# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True) 

st.sidebar.title('Select an application from below')

# app = st.sidebar.selectbox("SELECT",
#                            ("Predictions",
#                             "Visualization"))

@st.cache_data
def patient():
    patient = pd.read_csv('data/cohort_icu_readmission_30_I50.csv.gz', compression='gzip', error_bad_lines=False)
    return patient

@st.cache_data
def icd():
    icd = pd.read_csv('data/preproc_diag_icu.csv.gz', compression='gzip', error_bad_lines=False)
    return icd

x = patient()
y = icd()

def prediction(x,y):
    st.write('\n')
    data_load_state = st.text('Loading data...')
    patient = x
    icd = y

    # Divide the screen into two columns
    col1, col2 = st.columns(2)

# First column: Number of Gender with Heart Failure
    with col1:
        st.subheader('Number of Gender with Heart Failure')
        gender_counts = patient['gender'].value_counts()
        st.bar_chart(gender_counts)

    with col2:
        st.subheader('Number of Age\'s Patients with Heart Failure')
        age_counts = patient['Age'].value_counts()
        st.bar_chart(age_counts)

def visualization(x,y):
    st.write('\n')
    data_load_state = st.text('Loading data...')
    patient = x
    icd = y

    # Divide the screen into two columns
    col1, col2 = st.columns(2)

# First column: Number of Gender with Heart Failure
    with col1:
        st.subheader('Number of Gender with Heart Failure')
        gender_counts = patient['gender'].value_counts()
        st.bar_chart(gender_counts)


page = st.sidebar.selectbox('SELECT',['Predictions','Visualization']) 
if page == 'Predictions':
    prediction(x,y)
else:
    visualization(x,y)
