import pandas as pd
import numpy as np
import torch
import pickle
from sentence_transformers import SentenceTransformer, util

# Load dataset (Ensure you upload Coursera.csv to Colab)
df = pd.read_csv("Coursera.csv")

# Load SBERT model (lightweight but powerful)
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')


# Generate embeddings for course descriptions
df['Embeddings'] = df['Course Description'].apply(lambda x: model.encode(str(x), convert_to_tensor=True))

print("Embeddings generated successfully!")



#########above code for testing to see if bert model generates the embedding



# Load model once
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Compute embeddings and store in a list
embeddings = [model.encode(str(desc), convert_to_tensor=True) for desc in df['Course Description']]
df['Embeddings'] = embeddings

# Save to a pickle file
with open('course_embeddings.pkl', 'wb') as f:
    pickle.dump(df, f)

print("Embeddings saved!")

####HOW BERT MODEL IS USED TO CREATE PICKLE FILE