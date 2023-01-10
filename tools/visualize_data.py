import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def visualize(df):
    st.write(df)
    st.write(df.describe())
    st.text('Null values:')
    st.write(df.isnull().sum())

    plt.figure(1 , figsize = (15 , 6))
    n = 0 
    for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        n += 1
        plt.subplot(1 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.distplot(df[x] , bins = 15)
        plt.title('Distplot of {}'.format(x))
    st.pyplot(plt.show())

    # fig = plt.figure(figsize=(10, 4))
    # sns.pairplot(df, vars = ['Spending Score (1-100)', 'Annual Income (k$)', 'Age'], hue = "Gender")
    # st.pyplot(fig)


