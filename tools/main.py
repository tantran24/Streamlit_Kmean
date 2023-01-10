import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import streamlit as st
import plotly.express as px

sys.path.append('.')

import matplotlib.pyplot as plt
from visualize_data import visualize

def plot(df, data, label, centroids, options):
    u_labels = np.unique(label)
    

    if len(options) == 2: 
        plt.xlabel(options[0])
        plt.ylabel(options[1])
        for i in u_labels:
            plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)

        plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
        plt.legend()
        st.pyplot(plt.show())

    elif len(options) > 2:
        df["label"] = df["label"].astype(str)
        fig = px.scatter_3d(df, x=options[0], y=options[1], z=options[2], color='label')
        st.write(fig)
    
    


def train(df):
    clusters = st.slider('Choose number of clusters: ', 0, 20, 1)

    model = KMeans(n_clusters=clusters)

    options = st.multiselect(
    'Choose feature for clustering: ',
    ['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

    X = df[options].iloc[:,:].values

    label = model.fit_predict(X)
    centroids = model.cluster_centers_
    df['label'] = label

    st.header('Result')
    plot(df, X, label, centroids, options)

    

if __name__ == "__main__":
    data_path = './data/segmented_customers.csv'
    df = pd.read_csv(data_path)
    st.title('VISUALIZE DATA')
    visualize(df)
    st.title('TRAIN DATA')
    train(df)


