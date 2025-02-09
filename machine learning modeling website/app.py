import streamlit as st
import data_extraction
import data_insights
import data_process
import data_visualize
import classification_model
import regression_model
import association_model
import inferential_statistics_analysis

# Main Streamlit App
def main():
    st.sidebar.title("Machine Learning Modeling")
    options = st.sidebar.selectbox(
        "Choose an option",
        ["Home", "Upload Datasets", "Detail Information of Dataset","Data Visualizations","Inferential Statistics Analysis", "Data Preprocessing","Data Modeling"]
    )

    if options == "Home":
        st.title("Welcome to Machine Learning Modeling")
        
    elif options == "Upload Datasets":
        df = data_extraction.dataset_upload()
        if df is not None:
            st.session_state['df'] = df

    elif options == "Detail Information of Dataset":
        if 'df' in st.session_state:
            data_insights.detail_information(st.session_state['df'])
        else:
            st.warning("Please upload a dataset first in the 'Upload Datasets' section.")
    elif options == "Data Preprocessing":
        if 'df' in st.session_state:
            st.session_state['df'] = data_process.data_preprocessing(st.session_state['df'])
        else:
            st.warning("Please upload a dataset first in the 'Upload Datasets' section.")
    elif options == "Data Visualizations":
        if 'df' in st.session_state:
            data_visualize.Visualization(st.session_state['df'])
        else:
            st.warning("please upload a dataset first in the 'Upload Datasets' section.")
    elif options=="Inferential Statistics Analysis":
        if 'df' in st.session_state:
            inferential_statistics_analysis.inferential_statistics(st.session_state['df'])
        else:
            st.warning("please upload a dataset first in the 'Upload Datasets' section.")
    elif options=="Data Modeling":
        if 'df' in st.session_state:
            model_name=st.sidebar.selectbox("choose the type of modeling:",["Classification","Regression","Association"])
            if model_name=="Classification":
                classification_model.Modeling_for_classifications(st.session_state['df'])
            elif model_name=="Regression":
                regression_model.Modeling_for_regression(st.session_state['df'])
            elif model_name=="Association":
                association_model.association_modeling(st.session_state['df'])
                
    else:
        st.error("Error check the code")
if __name__ == "__main__":
    main()