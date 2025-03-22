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
