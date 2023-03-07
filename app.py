import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import folium_static
import folium
from datetime import datetime, time
import joblib
from tensorflow import keras
from io import BytesIO
import requests 

# Set Color Pallete
custom_colors = ["#ABC9FF", "#FFDEDE", "#FF8B8B", "#EB4747"]
sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)

data_url = "./testSample.csv"
xgb_pipe_url = "./pipeline/xgb_pipe.pkl"
preprocessor_url = "./pipeline/preprocessor.pkl"
dnn_url = "./pipeline/dnn"

st.title("FRAUD DETECTION WEB APP")
st.markdown("Advanced Data Science Capstone Project offered by IBM on Coursera")


@st.cache_data
def load_data():
    df = pd.read_csv(data_url, on_bad_lines='skip')
    df["dob"] = pd.DatetimeIndex(df["dob"])
    df["trans_date_trans_time"] = pd.DatetimeIndex(df["trans_date_trans_time"])
    df.drop(columns=["Unnamed: 0"], inplace=True)
    return df

@st.cache_resource
def load_model():
    # xgb_pipe = joblib.load(BytesIO(requests.get(xgb_pipe_url).content))
    # preprocessor = joblib.load(BytesIO(requests.get(preprocessor_url).content))
    xgb_pipe = joblib.load(xgb_pipe_url)
    preprocessor = joblib.load(preprocessor_url)
    dnn = keras.models.load_model(dnn_url)

    return xgb_pipe, preprocessor, dnn

def plot_cm(labels, predictions, p=0.5, clf=''):
    from sklearn.metrics import classification_report, confusion_matrix
    st.subheader(clf)
    # Plot Confusion Matrix
    predictions = predictions > p
    cm = confusion_matrix(labels, predictions)
    
    fig = plt.figure(figsize=(12,6))
    sns.heatmap(cm, annot=True, cmap=custom_colors, fmt="d")
    plt.title('Confusion matrix '+str(clf))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    st.pyplot(fig.figure)

    # Classification report
    st.text(classification_report(labels, predictions, digits=4))

def histograms():
    default = ['trans_date_trans_time',
                'category','amt',
                'city_pop','dob']
    
    select = ['trans_date_trans_time',
                'category','amt','gender',
                'street','city','state','zip','lat','long',
                'city_pop','job','dob','unix_time',
                'merch_lat','merch_long','is_fraud']
    
    col = st.multiselect("Columns:", select, default=default)

    num_plots = len(col)
    if num_plots == 0:
        fig = plt.figure(figsize=(16,5))
        plt.text(x=0.4, y=0.5, s="Please select columns!", fontdict={'size': 16})
        return fig

    if num_plots == 1:
        fig = plt.figure(figsize=(16,5))
        plt.tick_params(labelbottom=(False if c in ['category', 'street','city','state','zip', 'job'] else True))
        plt.hist(normal_sample[col], bins=50, color=custom_colors[0])
        plt.hist(fraud_sample[col], bins=50, color=custom_colors[-2])
        plt.legend(["Normal", "Fraud"], loc='upper right')
        plt.title(col[0])
        return fig
    
    fig, axs = plt.subplots(num_plots, figsize=(16,5*num_plots))
    for ax, c in zip(axs.ravel(), col):
        ax.tick_params(labelbottom=(False if c in ['category', 'street','city','state','zip', 'job'] else True))
        ax.hist(normal_sample[c], bins=50, color=custom_colors[0])
        ax.hist(fraud_sample[c], bins=50, color=custom_colors[-2])
        ax.legend(["Normal", "Fraud"], loc='upper right')
        ax.set_title(c)

    return fig

def map():
    longitude = df["long"].mean()
    latitude = df["lat"].mean()
    m = folium.Map(location=[latitude, longitude], zoom_start=4.2)

    normal_limit = min(normal_sample_nrows, 50)
    fraud_limit = min(fraud_sample_nrows, 20)

    coordinates = normal_sample[["lat", "long"]].drop_duplicates(keep="first").to_numpy()[:normal_limit]
    fraud_coordinates = fraud_sample[["lat", "long"]].drop_duplicates(keep="first").to_numpy()[:fraud_limit]

    for coor in coordinates:
        folium.Marker(coor, icon=folium.Icon(color='blue', icon='')).add_to(m)

    for fraud_coor in fraud_coordinates:
        folium.Marker(fraud_coor, icon=folium.Icon(color='red', icon='')).add_to(m)

    return folium_static(m)

def jointplot():
    select = ['amt', 'lat','long', 'city_pop','unix_time',
                'merch_lat','merch_long','is_fraud']
    
    with st.form("select box"):
        x = st.selectbox("X-axis", select)
        y = st.selectbox("Y-axis", select)
        submitted = st.form_submit_button("Submit")
        if submitted and x != y:
            fig = sns.jointplot(data=pd.concat([normal_sample, fraud_sample]), x=x, y=y, hue="is_fraud")
            return fig  
        else:
            fig = plt.figure(figsize=(16,5))
            plt.text(x=0.4, y=0.5, s="Please select 2 different columns!", fontdict={'size': 16})
            return fig 

def ETL(original_df):
    df = original_df.copy()

    df["hour"] = pd.DatetimeIndex(df["trans_date_trans_time"]).hour
    df["day_of_week"] = pd.DatetimeIndex(df["trans_date_trans_time"]).day_of_week

    df["dob"] = pd.DatetimeIndex(df["dob"]).year

    df['late_hour'] = df['hour'] >= 22
    df['early_hour'] = df['hour'] <= 3

    df['elderly'] = df['dob'] <= 1960
    df['young'] = df['dob'] >= 1990

    df = df[cols + ["is_fraud"]]

    return df


######## LOAD DATA ########
df = load_data()
normal = df[df["is_fraud"] == 0]
fraud = df[df["is_fraud"] == 1]

normal_sample_nrows = st.sidebar.slider("Number of normal transaction", 0, normal.shape[0]//10, step=10, value = normal.shape[0]//10)
fraud_sample_nrows = st.sidebar.slider("Number of fraud transaction", 0, fraud.shape[0], step=5, value=fraud.shape[0])

normal_sample = normal.sample(frac=normal_sample_nrows/normal.shape[0])
fraud_sample = fraud.sample(frac=fraud_sample_nrows/fraud.shape[0])


######### VISUALIZE #######
st.header("Explore the Dataset")

with st.expander("Show Raw Data"):
    st.write(df)

if normal_sample_nrows != 0 or fraud_sample_nrows != 0:
    with st.expander("Histograms"):
        st.subheader("Histograms")    
        st.pyplot(histograms())
    with st.expander("Map"):
        st.subheader("Transaction Positions")
        map()
    with st.expander("Features Jointplots"):
        st.subheader("Jointplots")
        st.pyplot(jointplot())


############ MODELS ###########
st.header("Models")

xgb_pipe, preprocessor, dnn = load_model()

cols = ['category', 'amt', 'gender', 'city_pop', 'lat', 'long', 
        'day_of_week', 'late_hour', 'early_hour', 'elderly', 'young']

if normal_sample_nrows != 0 or fraud_sample_nrows != 0:
    with st.expander("Predict and Evaluate on the dataset"):
        testing = pd.concat([normal_sample, fraud_sample])
        df1 = ETL(testing)
        X = df1[cols]
        y = df1["is_fraud"]
        y_pred = xgb_pipe.predict(X)
        plot_cm(y, y_pred, clf="XGBoost Classifier")

        X = preprocessor.transform(X)
        y_pred = dnn.predict(X)
        plot_cm(y, y_pred, clf="DeepNN Classifier")


with st.expander("Create your own transaction"):
    with st.form("Predict"):
        category = st.selectbox("Category", df['category'].unique())
        amt = st.number_input("Amount of Money", int(df["amt"].min()), int(df["amt"].max()), value=100, step=10)

        col1, col2 = st.columns(2)
        with col1:
            lat = st.slider("Latitude", int(df["lat"].min()), int(df["lat"].max()), value=45)
        with col2:
            long = st.slider("Longitude", int(df["long"].min()), int(df["long"].max()), value=-100)

        city_pop = st.number_input("City Population", int(df["city_pop"].min()), int(df["city_pop"].max()), value=1000000, step=20000)
       
        col1, col2 = st.columns(2)
        with col1:
            dob = st.date_input("Date of birth", min_value=datetime(1940, 1, 1))
        with col2:
            gender = st.radio("Gender", df['gender'].unique())

        col1, col2 = st.columns(2)
        with col1:
            trans_date = st.date_input("Transaction date")
        with col2:
            trans_time = st.time_input("Transaction time", time(0, 0))

        submit_button = st.form_submit_button(label='Predict')
        if submit_button:
            st.write("Predicting")

            day_of_week = trans_date.weekday()

            hour = trans_time.hour
            late_hour = hour >= 22
            early_hour = hour <= 3

            elderly = dob.year <= 1960
            young = dob.year >= 1990

            record = pd.DataFrame([[category, amt, gender, city_pop, lat, long, 
                  day_of_week, late_hour, early_hour, elderly, young]], columns=cols)
                
            result1 = xgb_pipe.predict_proba(record)
            x = preprocessor.transform(record)
            result2 = dnn.predict(x)

            result = pd.DataFrame({"Normal": [result1[0][0], 1-result2[0][0]], "Fraud": [result1[0][1], result2[0][0]]}, index=["XGBoost", "DeepNN"])
            
            result = result.style.applymap(lambda x: "background-color: #FF8B8B" if x>=0.5 else "background-color: #ABC9FF")
            st.write("Probability")
            st.write(result)


st.markdown("Created by Tri Cao Chanh 2023")