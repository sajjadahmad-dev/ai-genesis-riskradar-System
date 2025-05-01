import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
from transformers import pipeline
from groq import Groq
from serpapi import GoogleSearch

# ========== SETUP ==========
st.set_page_config(page_title="AI RiskRadar", layout="wide")
st.title("📡 AI RiskRadar - Real-Time Business Threat & Opportunity Analyzer")

# ========== API KEYS ==========
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# ========== PIPELINES & CLIENTS ==========
NER = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
groq_client = Groq(api_key=GROQ_API_KEY)

# ========== FUNCTIONS ==========

def fetch_news_data(query):
    search = GoogleSearch({
        "q": query,
        "num": 10,
        "api_key": SERPAPI_KEY
    })
    results = search.get_dict().get("organic_results", [])
    return [{
        "title": r["title"],
        "link": r["link"],
        "source": r.get("source", "Unknown")
    } for r in results]

def classify_risks(articles):
    tags = []
    for article in articles:
        entities = NER(article["title"])
        extracted = [e['word'] for e in entities if e['entity_group'] in ["ORG", "MISC", "PER"]]
        tags.extend(extracted)
    return pd.DataFrame(articles), tags

def generate_response(tags, topic):
    tag_text = ", ".join(tags)
    prompt = f"""
    Analyze the following tags: {tag_text} in the context of the topic: "{topic}".
    Identify any emerging business threats, market risks, or growth opportunities.
    Provide a short strategy summary for a startup or investor.
    """
    chat = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}]
    )
    return chat.choices[0].message.content.strip()

# ========== MAIN UI ==========
query = st.text_input("🔎 Enter a business, market, or topic to analyze:", "AI in Healthcare")

if st.button("Analyze"):
    with st.spinner("🚀 Scanning web for real-time risks and opportunities..."):
        articles = fetch_news_data(query)
        df, tags = classify_risks(articles)
        response = generate_response(tags, query)

    st.subheader("🧠 Strategic Insight from AI:")
    st.success(response)

    st.subheader("📰 Related News Articles:")
    st.dataframe(df[["title", "source", "link"]], use_container_width=True)

    if tags:
        tag_freq = pd.Series(tags).value_counts().reset_index()
        tag_freq.columns = ['Tag', 'Count']
        st.subheader("📊 Detected Entities & Tags")
        fig = px.bar(tag_freq, x='Tag', y='Count', title="Detected Tags (Entities, Keywords)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No significant entities or tags detected.")

# ========== FOOTER ==========
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Groq, HuggingFace, and SerpAPI.")
