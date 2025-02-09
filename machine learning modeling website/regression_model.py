import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def Modeling_for_regression(df):
    st.title("Regression Model Training")
    

    # Select target variable
    options = st.sidebar.selectbox("Choose the dependent column (Target Variable):", df.columns)

    try:
        # Splitting Features (X) and Target (y)
        X = df.drop(columns=[options])
        y = df[options]

        # Convert categorical variables if any
        X = pd.get_dummies(X, drop_first=True)

        # Standardization (required for Ridge, Lasso, ElasticNet)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Splitting data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Model Selection
        model_choice = st.sidebar.selectbox(
            "Choose a Regression Algorithm",
            ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net Regression",
             "Decision Tree Regression", "Random Forest Regression", "Gradient Boosting Regression", "XGBoost Regression"]
        )

        # Model Initialization
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Ridge Regression":
            alpha = st.sidebar.slider("Ridge Regularization Strength (alpha)", 0.01, 10.0, 1.0)
            model = Ridge(alpha=alpha)
        elif model_choice == "Lasso Regression":
            alpha = st.sidebar.slider("Lasso Regularization Strength (alpha)", 0.01, 10.0, 1.0)
            model = Lasso(alpha=alpha)
        elif model_choice == "Elastic Net Regression":
            alpha = st.sidebar.slider("Elastic Net Regularization Strength (alpha)", 0.01, 10.0, 1.0)
            l1_ratio = st.sidebar.slider("L1 Ratio (Elastic Net)", 0.0, 1.0, 0.5)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        elif model_choice == "Decision Tree Regression":
            max_depth = st.sidebar.slider("Max Depth (Decision Tree)", 2, 20, 5)
            model = DecisionTreeRegressor(max_depth=max_depth)
        elif model_choice == "Random Forest Regression":
            n_estimators = st.sidebar.slider("Number of Trees (Random Forest)", 10, 200, 100)
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        elif model_choice == "Gradient Boosting Regression":
            learning_rate = st.sidebar.slider("Learning Rate (Gradient Boosting)", 0.01, 0.5, 0.1)
            n_estimators = st.sidebar.slider("Number of Trees (Gradient Boosting)", 50, 200, 100)
            model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        elif model_choice == "XGBoost Regression":
            learning_rate = st.sidebar.slider("Learning Rate (XGBoost)", 0.01, 0.5, 0.1)
            n_estimators = st.sidebar.slider("Number of Trees (XGBoost)", 50, 200, 100)
            model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        # Train Model
        if st.sidebar.button("Train Model"):
            with st.spinner("Training the model..."):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Evaluation Metrics
                metrics = {
                    'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
                    "Mean Squared Error": mean_squared_error(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    "R2-Score": r2_score(y_test, y_pred)
                }

                st.success(f"Model training complete {model_choice}")

                # Display metrics
                st.subheader("Evaluation Metrics")
                for metric, value in metrics.items():
                    st.write(f"{metric}: {value:.4f}")

                # ================== New Graphs for Multiple Features ==================
                
                # **1. Residual Plot**
                st.subheader("Residual Plot (Actual - Predicted)")
                residuals = y_test - y_pred
                fig, ax = plt.subplots()
                sns.histplot(residuals, bins=30, kde=True, color='blue', ax=ax)
                ax.axvline(0, color='red', linestyle='dashed')  # Zero-error line
                bug=render_mpl_fig(fig)
                st.image(bug)

                # **2. Prediction vs Actual Scatter Plot**
                st.subheader("Prediction vs Actual Scatter Plot")
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
                ax.plot(y_test, y_test, color='red', linestyle='dashed')  # Ideal 45-degree line
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs. Predicted Values")
                bug=render_mpl_fig(fig)
                st.image(bug)

                # **3. Feature Importance (For Tree-Based Models)**
                if model_choice in ["Decision Tree Regression", "Random Forest Regression", "Gradient Boosting Regression", "XGBoost Regression"]:
                    st.subheader("Feature Importance")
                    feature_importance = model.feature_importances_
                    feature_names = X.columns
                    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
                    importance_df = importance_df.sort_values(by="Importance", ascending=False)

                    fig, ax = plt.subplots()
                    sns.barplot(x=importance_df["Importance"], y=importance_df["Feature"], palette="viridis", ax=ax)
                    ax.set_title("Feature Importance")
                    bug=render_mpl_fig(fig)
                    st.image(bug)
                    
    except:
        st.error(f"Error: Something went wrong. Please check the dataset and try again.**")

def render_mpl_fig(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format="png",bbox_inches="tight")
    buf.seek(0)
    return buf