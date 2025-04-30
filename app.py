
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
            {"title": "Category 4 Hurricane Makes Landfall", "snippet": "Winds up to 130 mph reported in coastal areas", "link": "#"},
            {"title": "Evacuations Underway", "snippet": "Over 1 million residents ordered to evacuate", "link": "#"},
            {"title": "Emergency Declared", "snippet": "National Guard deployed to affected regions", "link": "#"}
        ],
        "geo": {"lat": "27.6648", "lon": "-81.5158", "display_name": "Florida, USA"}
    },
    "earthquake": {
        "news": [
            {"title": "7.2 Magnitude Earthquake Strikes", "snippet": "Buildings collapsed in downtown area", "link": "#"},
            {"title": "Tsunami Warning Issued", "snippet": "Coastal residents urged to move to higher ground", "link": "#"},
            {"title": "International Aid Mobilized", "snippet": "UN sending emergency response teams", "link": "#"}
        ],
        "geo": {"lat": "35.6762", "lon": "139.6503", "display_name": "Tokyo, Japan"}
    }
}

# --- 🔐 Initialize API Clients with Secrets ---
try:
    groq = Groq(api_key=st.secrets.get("GROQ_API_KEY", "default-key"))
    SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "default-key")
except Exception as e:
    st.error(f"Failed to initialize API clients: {str(e)}")
    st.stop()

# --- 🧠 AI Model Initialization ---
@st.cache_resource
def load_ai_models():
    try:
        return {
            "disaster_clf": pipeline("text-classification", model="distilbert-base-uncased"),
            "ner": pipeline("ner", model="dslim/bert-base-NER"),
            "sentiment": pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
        }
    except Exception as e:
        st.error(f"Failed to load AI models: {str(e)}")
        return None

models = load_ai_models()
if models is None:
    st.stop()

# --- 🛰️ Data Fetching ---
def fetch_disaster_data(query, demo_mode=False):
    if demo_mode:
        disaster_type = random.choice(list(DEMO_DATA.keys()))
        return DEMO_DATA[disaster_type]
    
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
        return {"news": [], "geo": None}

# --- 🤖 AI Analysis Engine ---
def analyze_disaster(query, news_texts, geo_data):
    try:
        # Entity Recognition
        entities = models["ner"](" ".join(news_texts))
        locations = list({e['word'] for e in entities if e['entity'] in ['B-LOC','I-LOC']})
        
        # Disaster Classification with Groq
        disaster_prompt = f"""
        Analyze this disaster scenario:
        News Headlines: {news_texts[:2]}
        
        Respond with JSON containing:
        - "type": specific disaster type
        - "severity": 1-10 scale
        - "severity_rationale": brief explanation
        """
        
        disaster_analysis = groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": disaster_prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        ).choices[0].message.content
        
        disaster_analysis = json.loads(disaster_analysis)
        
        # Generate Response Plan
        response_prompt = f"""
        Generate response plan for:
        Disaster: {disaster_analysis['type']}
        Severity: {disaster_analysis['severity']}/10
        Locations: {locations}
        
        Provide JSON with:
        - "timeline": ["3 critical events"]
        - "actions": ["3 prioritized actions"]
        - "resources": ["3 needed resources"]
        """
        
        response_plan = groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": response_prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        ).choices[0].message.content
        
        return {
            **disaster_analysis,
            **json.loads(response_plan),
            "locations": locations,
            "geo": geo_data
        }
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

# --- 🎨 Streamlit UI ---
st.set_page_config(
    page_title="🚀 AI Disaster Response",
    layout="wide",
    page_icon="🌪️"
)

# Sidebar
with st.sidebar:
    st.image("https://i.imgur.com/JQ0w7wv.png", width=100)
    st.title("AI Genesis")
    demo_mode = st.checkbox("Demo Mode", value=True)
    st.markdown("---")
    st.markdown("### Models Used")
    st.markdown("- DistilBERT (Classification)")
    st.markdown("- BERT-NER (Location Extraction)")
    st.markdown("- Llama3-70B (Analysis)")

# Main Interface
st.title("🌪️ AI-Powered Disaster Response")
st.markdown("Real-time disaster intelligence with multi-model AI analysis")

# Input Section
query = st.text_input(
    "📍 Enter Disaster Location/Event:", 
    placeholder="e.g., Florida Hurricane 2025"
)

if st.button("🚀 Launch Analysis", type="primary"):
    with st.spinner("🛰️ Gathering intelligence..."):
        data = fetch_disaster_data(query, demo_mode=demo_mode)
        news_texts = [f"{n['title']}: {n.get('snippet', '')}" for n in data["news"]]
        
        if not news_texts:
            st.error("No valid news sources found.")
            st.stop()
            
        analysis = analyze_disaster(query, news_texts, data["geo"])
        if analysis is None:
            st.stop()
        
        # Results Dashboard
        st.success("✅ Analysis Complete!")
        
        # Severity Alert
        severity_color = "red" if analysis["severity"] > 7 else "orange" if analysis["severity"] > 4 else "green"
        st.markdown(f"""
        <div style="background:{severity_color}; padding:15px; border-radius:10px; color:white; margin-bottom:20px;">
            <h2 style="margin:0;">🚨 {analysis['type'].upper()}</h2>
            <h1 style="margin:0; text-align:center;">SEVERITY: {analysis['severity']}/10</h1>
            <p style="margin:0;"><i>{analysis['severity_rationale']}</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Map Visualization
        if analysis["geo"]:
            col1, col2 = st.columns([2, 1])
            with col1:
                m = folium.Map(
                    location=[float(analysis["geo"]["lat"]), float(analysis["geo"]["lon"])], 
                    zoom_start=7
                )
                folium.Marker(
                    [analysis["geo"]["lat"], analysis["geo"]["lon"]],
                    popup=f"<b>{query}</b>",
                    icon=folium.Icon(color=severity_color)
                ).add_to(m)
                folium_static(m)
            
            with col2:
                st.metric("📍 Location", analysis["geo"].get("display_name", "Unknown"))
                st.metric("🌍 Coordinates", f"{analysis['geo']['lat']}, {analysis['geo']['lon']}")
        
        # Timeline and Actions
        tab1, tab2 = st.tabs(["📅 Timeline", "🛠️ Response Plan"])
        
        with tab1:
            st.subheader("Critical Events")
            for event in analysis["timeline"]:
                st.markdown(f"⏱️ {event}")
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Priority Actions")
                for action in analysis["actions"]:
                    st.markdown(f"✅ {action}")
            with col2:
                st.subheader("Resources Needed")
                for resource in analysis["resources"]:
                    st.markdown(f"📦 {resource}")

# Footer
st.markdown("---")
st.markdown("Built for /execute: AI Genesis Hackathon")
