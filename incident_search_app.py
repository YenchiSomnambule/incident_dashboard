
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load data
df = pd.read_csv('Cleaned_Incidents.csv')
df = df.dropna(subset=['Incident_Description'])

# Embed incident descriptions
model = SentenceTransformer('all-MiniLM-L6-v2')
descriptions = df['Incident_Description'].tolist()
embeddings = model.encode(descriptions, show_progress_bar=True)

# Create FAISS index
dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# Streamlit UI
st.title("🔍 Incident Similarity Search Tool")
st.write("Enter a new incident or problem description below. This tool will search for the most similar past cases from your service logs.")

user_input = st.text_area("Describe the incident or issue:", height=150)

if st.button("Find Similar Incidents"):
    if not user_input.strip():
        st.warning("Please enter a description.")
    else:
        user_embedding = model.encode([user_input])
        D, I = index.search(np.array(user_embedding), k=5)
        st.subheader("Most Similar Incidents:")
        for rank, idx in enumerate(I[0]):
            case = df.iloc[idx]
            st.markdown(f"**#{rank + 1}: Incident ID {case['Incident_ID']}**")
            st.write(f"**Date:** {case['Date']}")
            st.write(f"**Department:** {case['Department']}")
            st.write(f"**Model:** {case['Model']} / **Sub-Assembly:** {case['Sub_Assembly']}")
            st.write(f"**Description:** {case['Incident_Description']}")
            st.markdown("---")
