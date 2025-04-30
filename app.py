import streamlit as st
import serpapi
import requests
from groq import Groq
from transformers import pipeline
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
from datetime import datetime
import json
import random

# --- 🌟 CONSTANTS ---
DEMO_DATA = {
    "hurricane": {
        "news": [
            {"title": "Category 4 Hurricane Makes Landfall", "snippet": "Winds up to 130 mph reported", "link": "#"},
            {"title": "Evacuations Underway", "snippet": "1 million residents ordered to evacuate", "link": "#"},
            {"title": "Emergency Declared", "snippet": "National Guard deployed", "link": "#"}
        ],
        "geo": {"lat": "27.6648", "lon": "-81.5158", "display_name": "Florida, USA"}
    }
}

# --- 🔐 Initialize API Clients with PROPER Secret Handling ---
try:
    # Method 1: Check if secrets exist (for Streamlit Cloud)
    if hasattr(st, 'secrets'):
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
        SERPAPI_KEY = st.secrets.get("SERPAPI_KEY")
    # Method 2: Fallback to environment variables
    else:
        import os
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
    
    # Method 3: Final fallback (for local testing only - remove before deployment)
    if not GROQ_API_KEY or not SERPAPI_KEY:
        GROQ_API_KEY = "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"
        SERPAPI_KEY = "39a147d2d97b7b81d98fe00e15a15edfa4e701f465c2f46df26ed534ef2cbd50"
    
    groq = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"API initialization failed: {str(e)}")
    st.stop()

# --- 🧠 AI Model Initialization ---
@st.cache_resource
def load_ai_models():
    try:
        return {
            "disaster_clf": pipeline("text-classification", model="distilbert-base-uncased"),
            "ner": pipeline("ner", model="dslim/bert-base-NER")
        }
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

models = load_ai_models()
if models is None:
    st.stop()

# --- 🛰️ Data Fetching ---
def fetch_disaster_data(query, demo_mode=False):
    if demo_mode:
        return DEMO_DATA["hurricane"]  # Use predefined demo data
    
    try:
        news = serpapi.search({
            "q": f"{query} disaster",
            "api_key": SERPAPI_KEY,
            "engine": "google_news",
            "num": 3
        }).get('news_results', [])[:3]
        
        geo_data = requests.get(
            f"https://nominatim.openstreetmap.org/search?q={query}&format=json"
        ).json()
        
        return {
            "news": [n for n in news if n.get('title')],
            "geo": geo_data[0] if geo_data else None
        }
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return None

# --- 🤖 AI Analysis Engine ---
def analyze_disaster(query, news_texts, geo_data):
    try:
        # Entity Recognition
        entities = models["ner"](" ".join(news_texts))
        locations = list({e['word'] for e in entities if e['entity'] in ['B-LOC','I-LOC']})
        
        # Disaster Analysis
        prompt = f"""
        Analyze this disaster:
        {query}
        News: {news_texts[:2]}
        
        Return JSON with:
        - "type": disaster type
        - "severity": 1-10
        - "actions": ["3 steps"]
        - "resources": ["3 items"]
        """
        
        response = groq.chat.completions.create(
            model="llama3-8b-8192",  # Using smaller model to conserve tokens
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        return {
            **json.loads(response.choices[0].message.content),
            "locations": locations,
            "geo": geo_data
        }
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

# --- 🎨 Streamlit UI ---
st.set_page_config(
    page_title="AI Disaster Response",
    layout="wide",
    page_icon="🌪️"
)

# Sidebar
with st.sidebar:
    st.title("AI Disaster Response")
    demo_mode = st.checkbox("Use Demo Mode", value=True)
    st.markdown("---")
    st.markdown("**Models:**")
    st.markdown("- DistilBERT")
    st.markdown("- BERT-NER")
    st.markdown("- Llama3")

# Main Interface
st.header("🌪️ Real-time Disaster Analysis")

query = st.text_input("Enter disaster location:", "Florida Hurricane")

if st.button("Analyze"):
    with st.spinner("Processing..."):
        data = fetch_disaster_data(query, demo_mode)
        if not data:
            st.error("Failed to fetch data")
            st.stop()
            
        news_texts = [f"{n['title']}: {n.get('snippet', '')}" for n in data["news"]]
        analysis = analyze_disaster(query, news_texts, data["geo"])
        
        if not analysis:
            st.stop()
        
        # Display Results
        st.subheader(f"{analysis['type'].upper()} DETECTED")
        
        # Severity Meter
        severity = analysis.get("severity", 5)
        st.progress(severity/10, text=f"Severity: {severity}/10")
        
        # Map
        if analysis["geo"]:
            m = folium.Map(
                location=[float(analysis["geo"]["lat"]), float(analysis["geo"]["lon"])],
                zoom_start=7
            )
            folium.Marker(
                [analysis["geo"]["lat"], analysis["geo"]["lon"]],
                tooltip=query
            ).add_to(m)
            folium_static(m)
        
        # Action Plan
        st.subheader("Action Plan")
        for action in analysis.get("actions", []):
            st.markdown(f"- {action}")
        
        # Resources
        st.subheader("Needed Resources")
        for resource in analysis.get("resources", []):
            st.markdown(f"- {resource}")

# Footer
st.markdown("---")
st.caption("Built for AI Genesis Hackathon")
