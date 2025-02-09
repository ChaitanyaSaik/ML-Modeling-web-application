import pandas as pd
import streamlit as st
import io
import matplotlib.pyplot as plt
import seaborn as sns 

def Visualization(df):
    st.title("Data Visualization Trends")

    if df is not None:
        options_for_graphs = [
            "Bar Chart", "Line Chart", "Scatter Plot", 
            "Histogram", "Count Plot", "Pie Chart",
            "Box Plot", "Violin Plot", "Heatmap", "Pairplot"
        ]
        
        graph_type = st.selectbox("Choose a graph type", options_for_graphs)

        if graph_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
            x_var = st.selectbox("Select X-axis variable", df.columns)
            y_var = st.selectbox("Select Y-axis variable", df.columns)
        elif graph_type in ["Histogram", "Count Plot", "Box Plot", "Violin Plot", "Pie Chart"]:
            x_var = st.selectbox("Select X-axis variable", df.columns)
            y_var = None
        elif graph_type in ["Heatmap", "Pairplot"]:
            x_var = None
            y_var = None

        if st.button("Generate Graph"):
            fig, ax = plt.subplots()

            if graph_type == "Bar Chart":
                sns.barplot(x=df[x_var], y=df[y_var], ax=ax)
            elif graph_type == "Line Chart":
                sns.lineplot(x=df[x_var], y=df[y_var], ax=ax)
            elif graph_type == "Scatter Plot":
                sns.scatterplot(x=df[x_var], y=df[y_var], ax=ax)
            elif graph_type == "Histogram":
                sns.histplot(df[x_var], kde=True, ax=ax)
            elif graph_type == "Count Plot":
                sns.countplot(x=df[x_var], ax=ax)
            elif graph_type == "Pie Chart":
                pie_data = df[x_var].value_counts()
                ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
                ax.set_title(f"Pie Chart of {x_var}")
            elif graph_type == "Box Plot":
                sns.boxplot(x=df[x_var], ax=ax)
            elif graph_type == "Violin Plot":
                sns.violinplot(x=df[x_var], ax=ax)
            elif graph_type == "Heatmap":
                fig, ax = plt.subplots(figsize=(10, 6))
                df_values=df.select_dtypes(include=['float64', 'int64'])
                sns.heatmap(df_values.corr(), annot=True, cmap="coolwarm", ax=ax)
            elif graph_type == "Pairplot":
                fig = sns.pairplot(df)

            buf = render_mpl_fig(fig)
            st.image(buf)

    else:
        st.error("No Dataset provided. Please upload a dataset first.")

def render_mpl_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf
