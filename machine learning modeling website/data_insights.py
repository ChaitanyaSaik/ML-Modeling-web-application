import zipfile
import pandas as pd
import streamlit as st
import io
import matplotlib.pyplot as plt
import seaborn as sns 

# Display detailed information about the dataset
def detail_information(df):
    st.title("Detail Information of Dataset")

    if df is not None:
        st.subheader("Head of the Dataset")
        st.dataframe(df.head())

        st.subheader("Tail of the Dataset")
        st.dataframe(df.tail())

        st.subheader("Info of the Dataset")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("Description of the Dataset")
        st.dataframe(df.describe())

        st.subheader("Null Values Count")
        st.write(df.isnull().sum())

        st.subheader("Duplicates Count")
        st.write(df.duplicated().sum())

        st.subheader("Target Variable Analysis")
        target_variable = st.selectbox("Choose the target variable", df.columns)
        if target_variable:
            st.write("Value Counts for Target Variable")
            value_counts = df[target_variable].value_counts()
            st.write(value_counts)

            fig, ax = plt.subplots()
            value_counts.plot(kind='bar', ax=ax)
            ax.set_title(f"Bar Chart of {target_variable}")
            ax.set_xlabel("Categories")
            ax.set_ylabel("Counts")
            buf=render_mpl_fig(fig)
            st.image(buf)
    else:
        st.error("No dataset provided. Please upload a dataset first.")

def render_mpl_fig(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format="png",bbox_inches="tight")
    buf.seek(0)
    return buf

