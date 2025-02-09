import zipfile
import pandas as pd
import streamlit as st
import io
import base64
import os


# Constants
OUTPUT_FOLDER = "uploaded_files"

# Ensure output folder exists
def create_output_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Save uploaded files or unzip them
def handle_uploaded_file(uploaded_file):
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1].lower()
    return file_name, file_extension

# Read datasets based on file type
def read_dataset(uploaded_file, file_extension):
    if file_extension == "csv":
        return pd.read_csv(uploaded_file)
    elif file_extension == "json":
        return pd.read_json(uploaded_file)
    elif file_extension in ["xls", "xlsx"]:
        return pd.read_excel(uploaded_file)
    else:
        return None

# Process ZIP file: extract and preview datasets
def process_zip_file(uploaded_file):
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(OUTPUT_FOLDER)
    extracted_files = os.listdir(OUTPUT_FOLDER)
    st.write("Extracted files:", extracted_files)

    for file in extracted_files:
        file_path = os.path.join(OUTPUT_FOLDER, file)
        file_extension = file.split('.')[-1].lower()
        try:
            df = read_dataset(file_path, file_extension)
            if df is not None:
                st.success(f"File {file} uploaded successfully!")
                st.write(f"Preview of {file}:")
                st.dataframe(df)
                return df
        except Exception as e:
            st.error(f"Error processing {file}: {e}")

    st.warning("No valid dataset found in the ZIP file.")
    return None

# Dataset Upload and Preview
def dataset_upload():
    st.title("Dataset Upload and Preview")
    create_output_folder()

    uploaded_file = st.file_uploader("Upload a file (CSV, JSON, Excel, or ZIP)", type=["csv", "json", "xlsx", "zip"])

    if uploaded_file:
        file_name, file_extension = handle_uploaded_file(uploaded_file)
        try:
            if file_extension in ["csv", "json", "xls", "xlsx"]:
                df = read_dataset(uploaded_file, file_extension)
                if df is not None:
                    st.success(f"{file_extension.upper()} file uploaded successfully!")
                    st.write("**Preview of the Dataset**")
                    st.dataframe(df)
                    return df
                else:
                    st.error("Unable to read the dataset.")
            elif file_extension == "zip":
                st.success("ZIP file uploaded successfully!")
                return process_zip_file(uploaded_file)
            else:
                st.error("Unsupported file type!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload a file to proceed.")
    return None

