
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

# Rules for potential causes and suggestions
def analyze_text(text):
    text = text.lower()
    cause = []
    suggestion = []

    if "missing" in text or "not included" in text:
        cause.append("Component may have been missed during picking or packing.")
        suggestion.append("Review packaging checklist and train staff on common oversight areas.")

    if "loose" in text or "wobbly" in text or "slipping" in text:
        cause.append("Improper tightening or missing fasteners during assembly.")
        suggestion.append("Review torque settings and ensure proper assembly verification steps.")

    if "damaged" in text or "bent" in text:
        cause.append("Likely shipping damage or weak protective materials.")
        suggestion.append("Improve packaging design and consider stronger protective materials.")

    if "incorrect" in text or "wrong" in text:
        cause.append("Incorrect item picked from inventory or mislabeled part.")
        suggestion.append("Audit inventory labeling and implement double-check at packing stage.")

    if not cause:
        cause.append("No specific cause identified.")
    if not suggestion:
        suggestion.append("Escalate to QA team for further investigation.")

    return cause, suggestion

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

            cause, suggestion = analyze_text(case['Incident_Description'])
            st.markdown(f"🛠 **Potential Cause:** {cause[0]}")
            st.markdown(f"✅ **Suggestion:** {suggestion[0]}")
            st.markdown("---")
