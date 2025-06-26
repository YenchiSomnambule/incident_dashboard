
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('enhanced_incident_data.csv')
issue_summary = pd.read_csv('issue_summary.csv')
model_summary = pd.read_csv('model_summary.csv')

# Embed descriptions
model = SentenceTransformer('all-MiniLM-L6-v2')
descriptions = df['Incident_Description'].tolist()
embeddings = model.encode(descriptions, show_progress_bar=False)
dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# Function to draw pie chart
def plt_pie(df, labels_col, values_col, title):
    fig, ax = plt.subplots()
    ax.pie(df[values_col], labels=df[labels_col], autopct="%1.1f%%", startangle=90, counterclock=False)
    ax.axis("equal")
    plt.title(title)
    return fig

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

# Streamlit layout
st.set_page_config(page_title="Incident Dashboard", layout="wide")
st.title("📊 Engineering Incident Dashboard + 🔍 Case Search")

tab1, tab2, tab3 = st.tabs(["📈 Insights", "🔍 Similarity Search", "📂 Raw Data"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Most Frequent Issue Types")
        st.pyplot(plt_pie(issue_summary, "Issue Category", "Count", "Issue Type Distribution"))

    with col2:
        st.subheader("Most Impacted Product Models")
        st.pyplot(plt_pie(model_summary.head(10), "Model", "Count", "Top 10 Models"))

with tab2:
    st.subheader("Find Similar Historical Incidents")
    user_input = st.text_area("Describe the issue or customer complaint:", height=150)

    if st.button("Search Similar Cases"):
        if not user_input.strip():
            st.warning("Please enter a description.")
        else:
            user_embedding = model.encode([user_input])
            D, I = index.search(np.array(user_embedding), k=5)

            st.subheader("Most Similar Past Incidents")
            for rank, idx in enumerate(I[0]):
                case = df.iloc[idx]
                st.markdown(f"### #{rank + 1}: Incident ID {case['Incident_ID']}")
                st.write(f"**Date:** {case['Date']}")
                st.write(f"**Department:** {case['Department']}")
                st.write(f"**Model:** {case['Model']} / **Sub-Assembly:** {case['Sub_Assembly']}")
                st.write(f"**Description:** {case['Incident_Description']}")

                cause, suggestion = analyze_text(case['Incident_Description'])
                st.markdown(f"🛠 **Potential Cause:** {cause[0]}")
                st.markdown(f"✅ **Suggestion:** {suggestion[0]}")
                st.markdown("---")

with tab3:
    st.subheader("Complete Incident Log")
    st.dataframe(df)
