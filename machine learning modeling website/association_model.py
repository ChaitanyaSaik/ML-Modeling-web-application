import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules

def association_modeling(df):
    try:
        st.title("Association Rule Mining for Heart Stroke Prediction")

        # Select Categorical Columns
        categorical_cols = st.multiselect("Select Categorical Columns for Association Analysis", df.columns)

        if not categorical_cols:
            st.warning("Please select at least one categorical column.")
            return

        # Run Apriori Algorithm to Find Frequent Itemsets
        try:
            min_support = st.slider("Select Minimum Support", min_value=0.01, max_value=0.5, step=0.01, value=0.1)
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        except Exception as e:
            st.error(f"Error in Apriori algorithm execution: {e}")
            return

        # Generate Association Rules
        try:
            min_confidence = st.slider("Select Minimum Confidence", min_value=0.1, max_value=1.0, step=0.1, value=0.5)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

            if rules.empty:
                st.warning("No significant rules found. Try lowering the support or confidence threshold.")
                return
        except Exception as e:
            st.error(f"Error in generating association rules: {e}")
            return

        # Display Frequent Itemsets
        st.subheader(" Frequent Itemsets")
        st.dataframe(frequent_itemsets)

        # Display Association Rules
        st.subheader("Association Rules")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

        # Convert sets to strings for better visualization
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

        #  Heatmap of Confidence Scores
        try:
            st.subheader("Heatmap of Confidence Scores")
            pivot_table = rules.pivot(index='antecedents', columns='consequents', values='confidence')
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error in generating heatmap: {e}")

        # Scatter Plot (Support vs Confidence)
        try:
            st.subheader(" Scatter Plot: Support vs Confidence")
            fig = go.Figure(data=[go.Scatter(
                x=rules['support'],
                y=rules['confidence'],
                mode='markers',
                marker=dict(size=8, color=rules['lift'], colorscale='Viridis', showscale=True),
                text=[f"Antecedents: {a}<br>Consequents: {c}" for a, c in zip(rules['antecedents'], rules['consequents'])]
            )])
            fig.update_layout(title="Support vs Confidence", xaxis_title="Support", yaxis_title="Confidence")
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error in generating scatter plot: {e}")

        #  Network Graph Visualization
        try:
            st.subheader("Network Graph of Association Rules")
            G = nx.DiGraph()
            for _, row in rules.iterrows():
                G.add_edge(row['antecedents'], row['consequents'], weight=row['lift'])

            plt.figure(figsize=(8,6))
            nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error in generating network graph: {e}")

        # **Conclusion of Association Model**
        st.subheader(" Conclusion")
        st.markdown("""
        - **Association Rule Mining** helps in identifying **frequent patterns and dependencies** between categorical features.
        - **Support**: Measures how frequently an itemset appears in the dataset.
        - **Confidence**: Probability that if antecedent occurs, the consequent will also occur.
        - **Lift**: Indicates the strength of the relationship between the antecedent and consequent.
        - **High-lift values** (>1) suggest strong associations.
        - **Useful for identifying risk factors and medical correlations** in heart stroke prediction.
        - **Visualization techniques** like **heatmaps, scatter plots, and network graphs** help interpret the patterns effectively.
        - **Business and Healthcare Applications**: This model can be used for **medical diagnosis**, **fraud detection**, and **recommendation systems**.
        """)

    except:
        st.error(f"An unexpected is errored")
