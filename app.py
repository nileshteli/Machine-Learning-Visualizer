import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

# Configure Streamlit page
st.set_page_config(page_title="📊 Machine Learning Visualizer", layout="wide")

# Gradient CSS background
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #00c9ff, #92fe9d, #6a11cb, #2575fc);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    color: #fff;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.block-container {
    background-color: rgba(0, 0, 0, 0.7);
    border-radius: 10px;
    padding: 2rem;
}

h1, h2, h3, h4 {
    color: #ffe066;
}
</style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1 style='text-align: center;'>📊 Machine Learning Visualizer</h1>", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("📁 - Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    if numeric_columns:
        st.subheader("📈 Data Visualization")
        col1, col2 = st.columns(2)

        graph_type = col1.selectbox("Graph Type", ["Line", "Scatter", "Bar", "Histogram", "Box", "Violin"])
        x_axis = col1.selectbox("X-axis", numeric_columns)
        y_axis = col2.selectbox("Y-axis", numeric_columns)

        fig, ax = plt.subplots()
        if graph_type == "Line":
            sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
        elif graph_type == "Scatter":
            sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
        elif graph_type == "Bar":
            sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)
        elif graph_type == "Histogram":
            sns.histplot(df[y_axis], kde=True, ax=ax)
        elif graph_type == "Box":
            sns.boxplot(data=df, x=x_axis, y=y_axis, ax=ax)
        elif graph_type == "Violin":
            sns.violinplot(data=df, x=x_axis, y=y_axis, ax=ax)

        st.pyplot(fig)

        st.subheader("🧠 Train a Machine Learning Model")
        target_column = st.selectbox("🎯 Select Target", numeric_columns)
        features = [col for col in numeric_columns if col != target_column]

        if st.button("🚀 Train Model"):
            df = df.dropna(subset=features + [target_column])
            X = df[features]
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if len(y.unique()) <= 10:
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.metric("Classification Accuracy", f"{acc:.2f}")
            else:
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                st.metric("R² Score", f"{r2:.2f}")
                fig2, ax2 = plt.subplots()
                sns.scatterplot(x=y_test, y=y_pred, ax=ax2)
                ax2.set_xlabel("Actual")
                ax2.set_ylabel("Predicted")
                st.pyplot(fig2)

# Sidebar Chatbot
with st.sidebar:
    st.markdown("### 🤖 Chatbot Assistant")
    question = st.text_input("Ask something:")

    if question:
        response = ""
        if "upload" in question.lower():
            response = "Use the CSV uploader on the main screen."
        elif "graph" in question.lower():
            response = "Choose graph type, X, and Y axes to visualize data."
        elif "model" in question.lower():
            response = "Select a target and click 'Train Model' to see accuracy."
        else:
            response = "I'm not sure. Try asking about 'upload', 'graph', or 'accuracy'."
        st.success(response)