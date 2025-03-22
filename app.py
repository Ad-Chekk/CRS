import streamlit as st
import pandas as pd
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity



import pandas as pd
import torch
import pickle

# Load the pickle file
with open("course_embeddings.pkl", "rb") as f:
    course_embeddings = pickle.load(f)

# Ensure it's a DataFrame
if not isinstance(course_embeddings, pd.DataFrame):
    raise TypeError(f"Expected a DataFrame, but got {type(course_embeddings)}")

# Print column names to verify available columns
print("Columns in DataFrame:", course_embeddings.columns)

# Fix column case issue
if 'Embeddings' in course_embeddings.columns:
    embeddings_column = 'Embeddings'
elif 'embeddings' in course_embeddings.columns:
    embeddings_column = 'embeddings'
else:
    raise KeyError("The DataFrame does not contain a column named 'Embeddings' or 'embeddings'.")

# Convert tensor values to lists safely
course_embeddings = course_embeddings.applymap(lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)

# Expand the 'Embeddings' column into separate columns if it's a list of tensors
try:
    expanded_embeddings = pd.DataFrame(course_embeddings[embeddings_column].to_list())
except Exception as e:
    raise ValueError(f"Error while expanding '{embeddings_column}'. Ensure it contains lists of numerical values.") from e

# Merge expanded embeddings back into the original DataFrame
course_embeddings = pd.concat([course_embeddings.drop(columns=[embeddings_column]), expanded_embeddings], axis=1)

# Display the first few rows
print(course_embeddings.head())

# Optionally, save the modified DataFrame
course_embeddings.to_csv("processed_embeddings.csv", index=False)



# Load the course embeddings DataFrame
with open("course_embeddings.pkl", "rb") as f:
    course_embeddings = pickle.load(f)

# Ensure the 'Embeddings' column exists and process it
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
    for idx, score in sorted_courses:
        recommended_course = course_embeddings.iloc[idx]["Course Name"]
        university = course_embeddings.iloc[idx]["University"]
        rating = course_embeddings.iloc[idx]["Course Rating"]
        
        st.write(f"**{recommended_course}** ({university}) - ⭐ {rating}")

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

# Run with: streamlit run app.py
