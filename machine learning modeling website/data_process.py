import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore


# Function for Outlier Detection
def detect_outliers(df, column, method):
    if method == "Z-Score":
        z_scores = np.abs(zscore(df[column]))
        outliers = df[z_scores > 3]
    elif method == "IQR":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    elif method == "Isolation Forest":
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        df["Outlier"] = iso_forest.fit_predict(df[[column]])
        outliers = df[df["Outlier"] == -1]
        df.drop(columns=["Outlier"], inplace=True)
    else:
        outliers = pd.DataFrame()
    
    return outliers

# Function for Outlier Removal
def remove_outliers(df, column, method):
    if method == "Z-Score":
        z_scores = np.abs(zscore(df[column]))
        df_clean = df[z_scores <= 3]
    elif method == "IQR":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]
    elif method == "Isolation Forest":
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        df["Outlier"] = iso_forest.fit_predict(df[[column]])
        df_clean = df[df["Outlier"] != -1].drop(columns=["Outlier"])
    else:
        df_clean = df.copy()
    
    return df_clean

def data_preprocessing(df):
    st.title("Data Preprocessing")

    if df is not None:
        # Display the dataset
        st.write("Dataset Preview:")
        st.dataframe(df.head(5))
        
        # Handling Missing Values
        st.subheader("Handling Missing Values")
        missing_categorical = st.selectbox("Choose handling method for categorical missing values", ["None", "Most Frequent", "Drop Rows"])
        missing_numerical = st.selectbox("Choose handling method for numerical missing values", ["None", "Mean", "Mode"])

        if st.button("Handle Missing Values"):
            if missing_categorical == "Most Frequent":
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    df[col].fillna(df[col].mode()[0],inplace=True)
                st.success("Handled missing values in categorical columns using Most Frequent value.")
            elif missing_categorical == "Drop Rows":
                df.dropna(subset=df.select_dtypes(include=['object']).columns)
                st.success("Dropped rows with missing values in categorical columns.")
            
            if missing_numerical == "Mean":
                numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numerical_cols:
                    df[col].fillna(df[col].mean(),inplace=True)
                st.success("Handled missing values in numerical columns using Mean.")
            elif missing_numerical == "Mode":
                numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numerical_cols:
                    df[col].fillna(df[col].mode()[0],inplace=True)
                st.success("Handled missing values in numerical columns using Mode.")

        # Drop Duplicates
        st.subheader("Drop Duplicate Rows")
        if st.button("Drop Duplicates"):
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                df.drop_duplicates(inplace=True)
                st.success(f"{duplicates} duplicate rows removed.")
            else:
                st.info("No duplicate rows found.")
        st.subheader("Outlier Detection and Removal ")
        if df is not None:
            column = st.selectbox("Select a Column for Outlier Detection", df.select_dtypes(include=np.number).columns)
            method = st.selectbox("Choose Outlier Detection Method", ["Z-Score", "IQR", "Isolation Forest"])
            if st.button("Detect Outliers"):
                outliers = detect_outliers(df, column, method)
                st.write(f"Detected Outliers ({len(outliers)} rows):", outliers)
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                sns.boxplot(y=df[column], ax=ax[0])
                ax[0].set_title("Original Data (Boxplot)")
                sns.boxplot(y=outliers[column], ax=ax[1])
                ax[1].set_title("Detected Outliers")
                st.pyplot(fig)


            if st.button("Remove Outliers"):
                df_clean = remove_outliers(df, column, method)
                st.write("Dataset After Removing Outliers:", df_clean.head(5))
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.boxplot(y=df_clean[column])
                ax.set_title("Data After Outlier Removal (Boxplot)")
                st.pyplot(fig)



        # Feature Engineering
        st.subheader("Feature Engineering")
        drop_columns = st.multiselect("Select columns to drop", df.columns)
        if st.button("Drop Columns"):
            if drop_columns:
                df.drop(columns=drop_columns, inplace=True)
                st.success(f"Dropped columns: {', '.join(drop_columns)}")

        # Normalization or Standardization
        st.subheader("Normalization or Standardization")
        scaling_method = st.selectbox("Choose scaling method for numerical values", ["None", "Normalization", "Standardization"])
        if st.button("Apply Scaling"):
            if scaling_method != "None":
                scaler = MinMaxScaler() if scaling_method == "Normalization" else StandardScaler()
                numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
                st.success(f"Applied {scaling_method} to numerical values.")

        # Categorical Encoding
        st.subheader("Categorical Values Encoding")
        categorical_scaling = st.selectbox("Choose encoding method", ["None", "Label Encoding", "One-Hot Encoding"])

        if categorical_scaling == "Label Encoding":
            categorical_cols = df.select_dtypes(include=['object']).columns
            if st.button("Apply Label Encoding"):
                if not categorical_cols.empty:
                    encoder = LabelEncoder()
                    for col in categorical_cols:
                        df[col] = encoder.fit_transform(df[col])
                    st.success("Categorical values encoded using Label Encoding.")

        elif categorical_scaling == "One-Hot Encoding":
            categorical_cols = df.select_dtypes(include=['object']).columns
            if st.button("Apply One-Hot Encoding"):
                if not categorical_cols.empty:
                    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
                    st.success("Categorical values encoded using One-Hot Encoding.")

        st.write("Preprocessing completed successfully!")
        st.dataframe(df.head(15))

        return df

    else:
        st.error("No dataset provided. Please upload a dataset first.")
        return None
