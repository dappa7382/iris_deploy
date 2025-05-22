import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import load_iris

# Load model
model = joblib.load('naive_bayes_model.pkl')

# Load iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['species'] = iris_df['target'].apply(lambda x: iris.target_names[x])

# Streamlit App
st.set_page_config(page_title="Iris Classifier", layout="centered")

# Sidebar navigation
page = st.sidebar.selectbox("Choose a page", ["Data Description", "Prediction", "About the Model"])

if page == "Data Description":
    st.title("Iris Dataset Description")
    st.write("""
        The **Iris dataset** is a classic dataset in machine learning and statistics. It contains 150 samples of iris flowers,
        with 4 features each:
        - Sepal length (cm)
        - Sepal width (cm)
        - Petal length (cm)
        - Petal width (cm)

        There are three classes of iris plants:
        - Iris-setosa
        - Iris-versicolor
        - Iris-virginica
    """)
    st.dataframe(iris_df.head(10))
    st.bar_chart(iris_df['species'].value_counts())

elif page == "Prediction":
    st.title("Predict Iris Species")
    st.write("Input the flower measurements below to predict the iris species.")

    sepal_length = st.slider("Sepal Length (cm)", float(iris_df["sepal length (cm)"].min()), float(iris_df["sepal length (cm)"].max()))
    sepal_width = st.slider("Sepal Width (cm)", float(iris_df["sepal width (cm)"].min()), float(iris_df["sepal width (cm)"].max()))
    petal_length = st.slider("Petal Length (cm)", float(iris_df["petal length (cm)"].min()), float(iris_df["petal length (cm)"].max()))
    petal_width = st.slider("Petal Width (cm)", float(iris_df["petal width (cm)"].min()), float(iris_df["petal width (cm)"].max()))

    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]  # list of lists

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.success(f"The predicted iris species is: **{iris.target_names[prediction]}**")

elif page == "About the Model":
    st.title("About the Model")
    st.write("""
        This model uses **Naive Bayes classification** to predict the species of iris flowers.

        - Naive Bayes is a probabilistic classifier based on Bayes' theorem.
        - It assumes independence between features.
        - It's efficient and performs well on small datasets like Iris.

        **Deployment:** This app is built with Streamlit, a fast way to build and share data apps in Python.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_dataset_scatterplot.svg", caption="Iris dataset visualization", use_column_width=True)
