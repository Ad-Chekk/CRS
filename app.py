import streamlit as st
import pandas as pd
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the course embeddings DataFrame
with open("course_embeddings.pkl", "rb") as f:
    course_embeddings = pickle.load(f)

# Ensure 'Embeddings' column exists and process it
if "Embeddings" in course_embeddings.columns:
    course_embeddings["Embeddings"] = course_embeddings["Embeddings"].apply(
        lambda x: x.tolist() if isinstance(x, torch.Tensor) else x
    )
    
    # Extract embeddings as a separate DataFrame
    embeddings_df = pd.DataFrame(course_embeddings["Embeddings"].to_list())

    # Drop original 'Embeddings' column and merge expanded embeddings
    course_embeddings = pd.concat([course_embeddings.drop(columns=["Embeddings"]), embeddings_df], axis=1)
else:
    st.error("The DataFrame does not contain a column named 'Embeddings'.")
    st.stop()

# Compute cosine similarity between course embeddings
similarity_matrix = cosine_similarity(embeddings_df)

# Streamlit UI
st.title("🎓 Course Recommendation System")

# Course selection dropdown
selected_course = st.selectbox("Select a course:", course_embeddings["Course Name"])

if selected_course:
    # Find the index of the selected course
    course_idx = course_embeddings[course_embeddings["Course Name"] == selected_course].index[0]

    # Get similarity scores for the selected course
    similarity_scores = list(enumerate(similarity_matrix[course_idx]))

    # Sort courses based on similarity scores (excluding itself)
    sorted_courses = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]  # Top 5 recommendations

    st.subheader("📌 Recommended Courses:")
    
    # Display recommended courses in a card format
    for idx, score in sorted_courses:
        recommended_course = course_embeddings.iloc[idx]["Course Name"]
        university = course_embeddings.iloc[idx]["University"]
        rating = course_embeddings.iloc[idx]["Course Rating"]
        description = course_embeddings.iloc[idx]["Course Description"]
        url = course_embeddings.iloc[idx]["Course URL"]
        
        with st.container():
            st.markdown(
                f"""
                <div style='padding: 10px; border-radius: 10px; background-color: #f4f4f4; margin-bottom: 10px;'>
                    <h4>{recommended_course} ({university})</h4>
                    <p><strong>⭐ Rating:</strong> {rating}</p>
                    <p><strong>Description:</strong> {description[:150]}...</p>
                    <a href='{url}' target='_blank'><button style='background-color:#4CAF50; color: white; border: none; padding: 5px 10px; border-radius: 5px;'>View Course</button></a>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Search functionality
st.subheader("🔍 Search Courses")
search_query = st.text_input("Enter course name:")
if search_query:
    filtered_courses = course_embeddings[
        course_embeddings["Course Name"].str.contains(search_query, case=False, na=False)
    ]
    st.dataframe(filtered_courses)

# Save processed data button
if st.button("💾 Save Processed Data"):
    course_embeddings.to_csv("processed_courses.csv", index=False)
    st.success("Processed data saved as 'processed_courses.csv'")
