import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load course data
@st.cache_data
def load_data():
    df = pd.read_csv("Coursera.csv")
    with open("course_embeddings.pkl", "rb") as f:
        course_embeddings = pickle.load(f)
    return df, np.array(course_embeddings)

df, course_embeddings = load_data()

# Compute similarity matrix
similarity_matrix = cosine_similarity(course_embeddings)

# Course Recommendation Function
def recommend_course(course_name, df, similarity_matrix, top_n=5):
    if course_name not in df["Course Name"].values:
        return ["Course not found in dataset."]
    
    idx = df[df["Course Name"] == course_name].index[0]
    similar_courses = list(enumerate(similarity_matrix[idx]))
    similar_courses = sorted(similar_courses, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in range(1, top_n + 1):  # Skip itself
        course_idx = similar_courses[i][0]
        recommendations.append(df.iloc[course_idx]["Course Name"])
    
    return recommendations

# Streamlit UI
st.title("📚 Course Recommendation System")

# Dropdown for course selection
course_name = st.selectbox("Select a course:", df["Course Name"].values)

# Recommend button
if st.button("Recommend Similar Courses"):
    recommendations = recommend_course(course_name, df, similarity_matrix)
    
    st.subheader("🔗 Recommended Courses:")
    for i, rec in enumerate(recommendations):
        st.write(f"{i+1}. {rec}")

