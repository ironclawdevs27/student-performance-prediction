import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time

# Importing Model
model = pickle.load(open('pipe.pkl', 'rb'))
st.sidebar.title("Welcome to Student Performance Predictor")
st.sidebar.header("Contents")

# About
page = st.sidebar.selectbox("Select Here", ['About', 'Input', 'Output'])

columns = ["gender","race/ethnicity","parental level of education","lunch",
           "test preparation course","math score","reading score","writing score"]


# Main Page
if page == 'About':
    st.title("Student Performance Predictor")
    st.markdown('''
    ### This model predicts the overall performance of a student using parameters entered by the user

    ### About Features
    - **Gender**: Female(0), Male(1)
    - **Race/Ethnicity**: Group-A(0), Group-B(1), Group-C(2), Group-D(3), Group-E(4)
    - **Parental Level of Education**: high school(0), some high school(1),some college(2), bachelor's degree(3), master's degree(4), associate's degree(5)
    - **Lunch**: Free(0), Standard(1)
    - **Test Prepartion Course**: Completed(1), None(0)
    - **Math Score**: 0 - 100
    - **Reading Score**: 0 - 100
    - **Writing Score**: 0 - 100
    '''
    )


# Page: Upload
elif page == 'Input':
    st.header("Enter Features")
    st.markdown('See the **About** section to know about feature values.')
    
    # Columns are created to display in grids!
    #-------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gender")
        gender = st.selectbox(label='0-F, 1-M', options=[0, 1],)

    with col2:
        st.subheader("Race")
        race = st.selectbox(label='A(0), B(1), C(2), D(3), E(4)', 
                            options=[0, 1, 2, 3, 4])

    # ------------------------------------------
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Parent's Education")
        level = st.selectbox(label='0-5', 
                             options=[0, 1, 2, 3, 4, 5])

    with col4:
        st.subheader("Lunch")
        lunch = st.selectbox(label='1- Standard, 0-Free', options=[0, 1])

    # ------------------------------------------
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Course Completed?")
        course = st.selectbox(label='0-No, 1-Yes', options=[0, 1])

    with col6:
        st.subheader("Math Score")
        math_score = st.number_input(label="Range: (0 - 100)", min_value=0, max_value=100)

    #--------------------------------------------
    col7, col8 = st.columns(2)
    with col7:
        st.subheader("Reading Score")
        reading_score = st.number_input(label="Range: 0 - 100", min_value=0, max_value=100)

    with col8:
        st.subheader("Writing Score")
        writing_score = st.number_input(label="Range: 0 to 100", min_value=0, max_value=100)

    def save():
        features = np.array([gender, race, level, lunch, course,  math_score, reading_score, writing_score])
        features = features.reshape(1,8)
        df = pd.DataFrame(features, columns=columns)
        df.to_csv("data.csv")

    st.button(label='Save', on_click=save)


# Page-3: Displaying Results!
elif page == 'Output':
    st.header("Output")
    st.markdown('''
        - Check the feature values with **Button-1**.
        - See prediction using **Button-2**.
    ''')
    df = pd.read_csv("data.csv")
    df.drop('Unnamed: 0', axis=1, inplace=True)

    def display_data():
        st.progress(value=0, text="In progress, please wait..")
        time.sleep(1)
        st.dataframe(df)

    def predict():
        percentage = model.predict(df)
        st.progress(value=1, text="Running...")
        time.sleep(1)
        st.subheader(f"Predicted Percentage: {np.round(percentage[0], decimals=2)} %")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Features")
        st.button(label='Button-1', on_click=display_data)

    with col2:
        st.subheader("Output")
        st.button(label='Button-2', on_click=predict)
        