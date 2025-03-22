import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load precomputed embeddings
with open('course_embeddings.pkl', 'rb') as f:
    df = pickle.load(f)

# Load the same model used for embeddings
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Function to recommend courses
def recommend_courses(course_name, top_n=5):
    if course_name not in df['Course Name'].values:
        return []
    
    idx = df[df['Course Name'] == course_name].index[0]
    query_embedding = df.loc[idx, 'Embeddings']
    
    similarities = [util.cos_sim(query_embedding, emb).item() for emb in df['Embeddings']]
    sorted_indices = np.argsort(similarities)[::-1][1:top_n+1]  # Exclude itself
    
    return df.iloc[sorted_indices][['Course Name', 'Course Description', 'Course Rating', 'Course URL']]

# Streamlit UI
st.title("📚 Course Recommendation System")
st.write("Enter a course name to get recommendations based on similarity.")

course_name = st.text_input("Enter a Course Name:")
if st.button("Recommend"):
    if course_name:
        recommendations = recommend_courses(course_name)
        if not recommendations.empty:
            st.write("### Recommended Courses:")
            for _, row in recommendations.iterrows():
                st.write(f"**{row['Course Name']}**")
                st.write(f"🔹 {row['Course Description'][:150]}...")
                st.write(f"⭐ Rating: {row['Course Rating']}")
                st.write(f"🔗 [Course Link]({row['Course URL']})")
                st.markdown("---")
        else:
            st.error("No recommendations found. Try another course!")
