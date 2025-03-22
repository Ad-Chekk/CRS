import streamlit as st
import pandas as pd
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Login Page", page_icon=":bar_chart:", layout="centered")
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://cdn.pixabay.com/photo/2016/09/05/15/03/candle-1646765_1280.jpg");
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)




# Load course embeddings
with open("course_embeddings.pkl", "rb") as f:
    course_embeddings = pickle.load(f)

# Process embeddings
if "Embeddings" in course_embeddings.columns:
    course_embeddings["Embeddings"] = course_embeddings["Embeddings"].apply(
        lambda x: x.tolist() if isinstance(x, torch.Tensor) else x
    )
    embeddings_df = pd.DataFrame(course_embeddings["Embeddings"].to_list())
    course_embeddings = pd.concat([course_embeddings.drop(columns=["Embeddings"]), embeddings_df], axis=1)
else:
    st.error("The DataFrame does not contain a column named 'Embeddings'.")
    st.stop()

# Compute cosine similarity
similarity_matrix = cosine_similarity(embeddings_df)

# Streamlit UI
st.title("🎓 Course Recommendation System")

# Select a course
dropdown_style = """
    <style>
        div[data-baseweb="select"] > div {
            background-color: #444;
            color: white;
        }
    </style>
"""
st.markdown(dropdown_style, unsafe_allow_html=True)
selected_course = st.selectbox("Select a course:", course_embeddings["Course Name"])

if selected_course:
    course_idx = course_embeddings[course_embeddings["Course Name"] == selected_course].index[0]
    similarity_scores = list(enumerate(similarity_matrix[course_idx]))
    sorted_courses = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]

    st.subheader("📌 Recommended Courses:")
    
    for idx, score in sorted_courses:
        recommended_course = course_embeddings.iloc[idx]
        course_name = recommended_course["Course Name"]
        description = recommended_course["Course Description"][:150] + "..."  # Truncate description
        rating = recommended_course["Course Rating"]
        url = recommended_course["Course URL"]
        
        card_html = f"""
        <div style="background: rgba(0, 0, 0, 0.6); padding: 15px; border-radius: 10px; margin: 10px 0; color: white;">
            <h3 style="margin-bottom: 5px;">{course_name}</h3>
            <p style="font-size: 14px;">{description}</p>
            <p><b>⭐ Rating:</b> {rating}</p>
            <a href="{url}" target="_blank" style="text-decoration: none;">
                <button style="background: #ff4b4b; color: white; padding: 7px 15px; border: none; border-radius: 5px; cursor: pointer;">
                    View Course
                </button>
            </a>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

# Search courses
st.subheader("🔍 Search Courses")
search_query = st.text_input("Enter course name:")
if search_query:
    filtered_courses = course_embeddings[course_embeddings["Course Name"].str.contains(search_query, case=False, na=False)]
    st.dataframe(filtered_courses)

# Save processed data button
if st.button("💾 Save Processed Data"):
    course_embeddings.to_csv("processed_courses.csv", index=False)
    st.success("Processed data saved as 'processed_courses.csv'")
