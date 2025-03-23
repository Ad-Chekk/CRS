import streamlit as st
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Course Recommendation", page_icon=":books:", layout="wide")

# Background Image
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

col1, col2 = st.columns([2, 1])

with col1:
    selected_course = st.selectbox("Select a course:", course_embeddings["Course Name"], index=0)
    if selected_course:
        course_idx = course_embeddings[course_embeddings["Course Name"] == selected_course].index[0]
        similarity_scores = list(enumerate(similarity_matrix[course_idx]))
        sorted_courses = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]
        
        st.subheader("📌 Recommended Courses:")
        recommendations = []
        
        for idx, score in sorted_courses:
            recommended_course = course_embeddings.iloc[idx]
            course_name = recommended_course["Course Name"]
            description = recommended_course["Course Description"][:150] + "..."
            rating = recommended_course["Course Rating"]
            url = recommended_course["Course URL"]
            recommendations.append((course_name, rating))
            
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
        
        # Convert recommendations to DataFrame for visualization
        rec_df = pd.DataFrame(recommendations, columns=["Course Name", "Rating"])

with col2:
    # Bar Chart - Top Rated Recommendations
    st.subheader("📊 Top Rated Recommended Courses")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(y=rec_df["Course Name"], x=rec_df["Rating"], palette="coolwarm", ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Course Name")
    st.pyplot(fig)

    # Pie Chart - Course Rating Distribution
    st.subheader("📊 Rating Distribution")
    fig2, ax2 = plt.subplots()
    course_embeddings["Course Rating"].value_counts().plot.pie(autopct="%1.1f%%", cmap="coolwarm", ax=ax2)
    ax2.set_ylabel("")
    st.pyplot(fig2)

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
