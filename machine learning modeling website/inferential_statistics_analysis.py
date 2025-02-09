import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import io

def inferential_statistics(df):
    st.title("Inferential Statistics Testing")

    # Select numerical columns
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numerical_columns) < 2:
        st.warning("The dataset must contain at least two numerical columns.")
        return

    col1 = st.selectbox("Select First Column:", numerical_columns)
    col2 = st.selectbox("Select Second Column:", numerical_columns)

    # Select test type
    test_type = st.selectbox("Choose a Statistical Test:", 
                             ["T-Test", "A/B Testing", "ANOVA", "Confidence Interval","Chi-Square","Correlation"])
    
    if st.button("Run Test"):
        # Initialize conclusion text
        conclusion = ""

        if test_type == "T-Test":
            # Perform Independent T-test
            t_stat, p_val = stats.ttest_ind(df[col1].dropna(), df[col2].dropna(), equal_var=False)
            st.write(f"**T-Test Results:**\nT-Statistic = {t_stat:.3f}, P-Value = {p_val:.5f}")

            # Plot Distribution (Fixed shade → fill)
            plt.figure(figsize=(8, 5))
            sns.kdeplot(df[col1], label=col1, fill=True)  # Fixed
            sns.kdeplot(df[col2], label=col2, fill=True)  # Fixed
            plt.title("T-Test Distribution Comparison")
            plt.legend()
            bug = render_mpl_fig(plt)
            st.image(bug)

            # Conclusion for T-Test
            if p_val < 0.05:
                conclusion += f"\nThe T-Test indicates a significant difference between {col1} and {col2} (p < 0.05)."
            else:
                conclusion += f"\nThe T-Test shows no significant difference between {col1} and {col2} (p > 0.05)."

        elif test_type == "A/B Testing":
            # A/B Testing (Chi-square test)
            observed = pd.crosstab(df[col1], df[col2])
            chi2, p_val, dof, expected = stats.chi2_contingency(observed)
            st.write(f"**A/B Testing Results:**\nChi-Square = {chi2:.3f}, P-Value = {p_val:.5f}")

            # Bar Plot for A/B Testing
            plt.figure(figsize=(8, 5))
            observed.plot(kind="bar", stacked=True)
            plt.title("A/B Test - Category Distribution")
            bug = render_mpl_fig(plt)
            st.image(bug)

            # Conclusion for A/B Testing
            if p_val < 0.05:
                conclusion += f"\nThe A/B Test shows a significant association between {col1} and {col2} (p < 0.05)."
            else:
                conclusion += f"\nThe A/B Test shows no significant association between {col1} and {col2} (p > 0.05)."

        elif test_type == "ANOVA":
            # Perform One-Way ANOVA
            groups = [df[col1][df[col2] == category].dropna() for category in df[col2].unique()]
            f_stat, p_val = stats.f_oneway(*groups)
            st.write(f"**ANOVA Results:**\nF-Statistic = {f_stat:.3f}, P-Value = {p_val:.5f}")

            # Box Plot for Group Distribution
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=df[col2], y=df[col1])
            plt.title("ANOVA - Group Comparisons")
            bug = render_mpl_fig(plt)
            st.image(bug)

            # Conclusion for ANOVA
            if p_val < 0.05:
                conclusion += f"\nANOVA shows significant differences between the groups of {col1} based on {col2} (p < 0.05)."
            else:
                conclusion += f"\nANOVA shows no significant differences between the groups of {col1} based on {col2} (p > 0.05)."

        elif test_type == "Confidence Interval":
            # Confidence Interval Calculation
            mean_val = np.mean(df[col1])
            std_err = stats.sem(df[col1].dropna())
            ci_low, ci_high = stats.t.interval(0.95, len(df[col1])-1, loc=mean_val, scale=std_err)
            st.write(f"**95% Confidence Interval for {col1}:**\nLower Bound = {ci_low:.3f}, Upper Bound = {ci_high:.3f}")

            # Confidence Interval Plot
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col1], bins=20, kde=True)
            plt.axvline(ci_low, color="red", linestyle="dashed", label="Lower Bound")
            plt.axvline(ci_high, color="green", linestyle="dashed", label="Upper Bound")
            plt.title(f"Confidence Interval for {col1}")
            plt.legend()
            bug = render_mpl_fig(plt)
            st.image(bug)

            # Conclusion for Confidence Interval
            conclusion += f"\nThe 95% Confidence Interval for {col1} is between {ci_low:.3f} and {ci_high:.3f}."

        elif stat_test == "Chi-Square":
            cat1 = st.selectbox("Select First Categorical Variable", df.select_dtypes(include=['object']).columns)
            cat2 = st.selectbox("Select Second Categorical Variable", df.select_dtypes(include=['object']).columns)
            if cat1 != cat2:
                # Perform Chi-Square Test
                contingency_table = pd.crosstab(df[cat1], df[cat2])
                chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                st.write(f"📊 **Chi-Square Test Results:**")
                st.write(f"Chi-Square Statistic: {chi2_stat:.3f}, P-Value: {p_value:.5f}")
                if p_value < 0.05:
                    st.success("✅ There is a statistically significant association between the variables!")
                else:
                    st.warning("⚠️ No significant association found.")

        elif stat_test == "Correlation":
            corr_type = st.selectbox("Choose correlation type", ["Pearson", "Spearman", "Kendall"])
            num1 = st.selectbox("Select First Numerical Variable", df.select_dtypes(include=['float64', 'int64']).columns)
            num2 = st.selectbox("Select Second Numerical Variable", df.select_dtypes(include=['float64', 'int64']).columns)
            if num1 != num2:
                # Compute correlation
                if corr_type == "Pearson":
                    corr, p_value = stats.pearsonr(df[num1], df[num2])
                elif corr_type == "Spearman":
                    corr, p_value = stats.spearmanr(df[num1], df[num2])
                else:
                    corr, p_value = stats.kendalltau(df[num1], df[num2])
                st.write(f"📊 **{corr_type} Correlation Results:**")
                st.write(f"Correlation Coefficient: {corr:.3f}, P-Value: {p_value:.5f}")
                if p_value < 0.05:
                    st.success("✅ The correlation is statistically significant!")
                else:
                    st.warning("⚠️ No significant correlation found.")

# Function to render Matplotlib figures in Streamlit
def render_mpl_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

