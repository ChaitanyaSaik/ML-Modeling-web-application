import pandas as pd
import streamlit as st
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB

def oversampling(X_train, y_train):
    try:
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        return X_train_resampled, y_train_resampled
    except:
        st.error("Error in oversampling, please check it.")
        return None, None

def render_mpl_fig(fig):
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return buf
    except:
        st.error("Error rendering plot, please check it.")
        return None

def Modeling_for_classifications(data):
    try:
        st.title("Data Modeling")
        options = st.sidebar.selectbox("Select dependent column:", data.columns)
        X = data.drop(options, axis=1)
        y = data[options]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        model_choice = st.sidebar.selectbox(
            "Choose an Algorithm",
            ("Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "SVC", "KNN", "GradientBoostingClassifier",
             "GaussianNB", "MultinomialNB", "BernoulliNB", "ComplementNB", "CategoricalNB")
        )    
        
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "XGBoost":
            model = XGBClassifier(eval_metric='logloss')
        elif model_choice == "KNN":
            model = KNeighborsClassifier()
        elif model_choice == "SVC":
            model = SVC(class_weight='balanced', probability=True)
        elif model_choice == "GradientBoostingClassifier":
            model = GradientBoostingClassifier()
        elif model_choice == "GaussianNB":
            model = GaussianNB()
        elif model_choice == "MultinomialNB":
            model = MultinomialNB()
        elif model_choice == "BernoulliNB":
            model = BernoulliNB()
        elif model_choice == "ComplementNB":
            model = ComplementNB()
        elif model_choice == "CategoricalNB":
            model = CategoricalNB()
        
        if st.sidebar.button("Train Model"):
            with st.spinner("Training the model..."):
                X_train, y_train = oversampling(X_train, y_train)
                if X_train is not None and y_train is not None:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    metrics = {
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred),
                        'Recall': recall_score(y_test, y_pred),
                        'F1 Score': f1_score(y_test, y_pred),
                        'ROC-AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
                    }
                    
                    st.success(f"Model training completed for {model_choice}")
                    
                    st.subheader("Evaluation Metrics")
                    for metric, value in metrics.items():
                        st.write(f"{metric}: {value:.4f}" if value is not None else f"{metric}: N/A")

                    st.subheader(f"Confusion Matix for {model_choice}")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                    plt.title(f"Confusion Matrix = {model_choice}")
                    plt.xlabel("Predicted Label")
                    plt.ylabel("True Label")
                    buf = render_mpl_fig(plt)
                    if buf:
                        st.image(buf)
                    
                    if y_proba is not None:
                        st.subheader(f"Roc Curve for {model_choice}")
                        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                        plt.figure(figsize=(8, 6))
                        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['ROC-AUC']:.4f})")
                        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.title(f"ROC Curve - {model_choice}")
                        plt.legend(loc="lower right")
                        buf = render_mpl_fig(plt)
                        if buf:
                            st.image(buf)
                    
                    # Create sigmoid graph for Logistic Regression (only for Logistic Regression)
                    if model_choice == "Logistic Regression":
                        st.subheader(f"Threshold graph for logistic Regression")
                        # Set a custom threshold (e.g., 0.5) and make predictions
                        threshold = 0.5
                        Y_pred_threshold = (y_proba >= threshold).astype(int)
                        # Plotting the sigmoid function to visualize probabilities and threshold
                        x_values = np.linspace(-5, 5, 100)
                        sigmoid = 1 / (1 + np.exp(-x_values))
                        plt.figure(figsize=(10, 6))
                        plt.plot(x_values, sigmoid, label="Sigmoid Function", color="blue")
                        plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
                        plt.scatter(model.decision_function(X_test), y_proba, c=y_test, cmap='coolwarm', alpha=0.7, edgecolors='k')
                        plt.xlabel("Model Decision Function")
                        plt.ylabel("Predicted Probability")
                        plt.title(f"Sigmoid Curve with {options} Probabilities and Threshold")
                        plt.legend()
                        bug=render_mpl_fig(plt)
                        if bug:
                            st.image(bug)
                    else:
                        st.write(" ")
                        
                    # Feature importance bar graph 
                    if model_choice == "Random Forest":
                        st.subheader(f"Feature Importance in {options} prediction")
                        feature_importance=model.feature_importances_
                        sorted_idx=np.argsort(feature_importance)

                        plt.figure(figsize=(8,5))
                        plt.barh(X.columns[sorted_idx],feature_importance[sorted_idx],color="skyblue")
                        plt.xlabel("Feature Importance")
                        plt.ylabel("features")
                        plt.title(f"Feature Importance in {options} prediction")
                        bug=render_mpl_fig(plt)
                        if bug:
                            st.image(bug)
                    elif model_choice == "XGBoost":
                        st.subheader(f"Feature Importance in {options}prediction")
                        feature_importance=model.feature_importances_
                        sorted_idx=np.argsort(feature_importance)

                        plt.figure(figsize=(8,5))
                        plt.barh(X.columns[sorted_idx],feature_importance[sorted_idx],color="skyblue")
                        plt.xlabel("Feature Importance")
                        plt.ylabel("features")
                        plt.title(f"Feature Importance in {options} prediction")
                        bug=render_mpl_fig(plt)
                        if bug:
                            st.image(bug)
                    else:
                        st.write(" ")
                else:
                    st.warning("Oversampling failed. Cannot train the model.")
        else:
            st.warning("Please choose a correct dependent column for modeling.")
    except:
        st.warning("An error occurred in the code, please check the dataset.")
