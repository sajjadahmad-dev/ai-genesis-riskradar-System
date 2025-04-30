import streamlit as st
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
import os

# Page config
st.set_page_config(
    page_title="🚀 AI Genesis: Disaster Response",
    layout="wide",
    page_icon="🌪️",
    initial_sidebar_state="expanded"
)

# Constants
DEMO_DATA = {
    "hurricane": {
        "news": [
            {"title": "Category 4 Hurricane Makes Landfall", "snippet": "Winds up to 130 mph reported in coastal areas", "link": "https://www.floridadisaster.org"},
            {"title": "Evacuations Underway", "snippet": "Over 1 million residents ordered to evacuate", "link": "https://www.fema.gov"},
            {"title": "Emergency Declared", "snippet": "National Guard deployed to affected regions", "link": "https://www.weather.gov"}
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

@st.cache_resource
def load_ai_models():
    return {
        "disaster_clf": pipeline("text-classification", model="distilbert-base-uncased"),
        "ner": pipeline("ner", model="dslim/bert-base-NER"),
        "sentiment": pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
    }

models = load_ai_models()
groq = Groq(api_key=os.getenv("GROQ_API_KEY", "your_groq_api_key_here"))

def fetch_disaster_data(query, demo_mode=False):
    if demo_mode:
        disaster_type = random.choice(list(DEMO_DATA.keys()))
        return DEMO_DATA[disaster_type]
    try:
        serpapi_key = os.getenv("SERPAPI_KEY", "your_serpapi_key_here")
        url = f"https://serpapi.com/search.json?q={query}+disaster&engine=google_news&num=3&api_key={serpapi_key}"
        resp = requests.get(url).json()
        news = resp.get("news_results", [])[:3]
        geo_data = requests.get(
            f"https://nominatim.openstreetmap.org/search?q={query}&format=json",
            headers={"User-Agent": "AI-Genesis-Hackathon"}
        ).json()
        if not news:
            raise ValueError("No news results")
        return {
            "news": news,
            "geo": geo_data[0] if geo_data else None
        }
    except Exception as e:
        st.error(f"Error fetching live data: {e}. Showing demo data.")
        return DEMO_DATA["hurricane"]

def analyze_disaster(query, news_texts, geo_data):
    try:
        entities = models["ner"](" ".join(news_texts))
        locations = list({e['word'] for e in entities if 'LOC' in e['entity']})
    except:
        locations = []

    disaster_prompt = f"""
    Analyze this disaster scenario:
    News Headlines: {news_texts[:2]}
    Respond in JSON:
    {{
        "type": "disaster type",
        "severity": 1-10,
        "severity_rationale": "short explanation"
    }}
    """
    try:
        response = groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": disaster_prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        ).choices[0].message.content
        disaster_data = json.loads(response)
    except:
        disaster_data = {
            "type": "Hurricane",
            "severity": 9,
            "severity_rationale": "Major landfall with high wind speeds."
        }

    response_prompt = f"""
    Plan response for disaster type: {disaster_data['type']}, Severity: {disaster_data['severity']}.
    Locations: {locations}
    Return JSON:
    {{
        "timeline": ["YYYY-MM-DD HH:MM:SS event", ...],
        "actions": ["step1", "step2", "step3"],
        "resources": ["res1", "res2", "res3"],
        "sentiment": "public sentiment"
    }}
    """
    try:
        plan_response = groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": response_prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        ).choices[0].message.content
        plan = json.loads(plan_response)
    except:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plan = {
            "timeline": [f"{now}: Landfall", f"{now}: Emergency response", f"{now}: Recovery starts"],
            "actions": ["Evacuate", "Rescue ops", "Medical aid"],
            "resources": ["Food", "Water", "Medical kits"],
            "sentiment": "Concerned"
        }

    try:
        sentiment = models["sentiment"](" ".join(news_texts[:3]))[0]
    except:
        sentiment = {"label": "Neutral", "score": 0.5}

    return {
        **disaster_data,
        **plan,
        "locations": locations,
        "geo": geo_data,
        "sentiment_label": sentiment["label"],
        "sentiment_score": sentiment["score"]
    }

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/5f/Hurricane_Laura_2020-08-27_0610Z.jpg", use_column_width=True)
    st.title("AI Genesis")
    st.markdown("**LabLab AI Hackathon Entry**")
    demo_mode = st.checkbox("Demo Mode", True)
    st.markdown("---")
    st.markdown("### Models Used")
    st.markdown("- DistilBERT\n- BERT-NER\n- LLaMA 3 (Groq)")
    st.markdown("---")
    st.markdown("Created with ❤️")

# Main
st.title("🌪️ AI-Powered Disaster Response")
st.markdown("Real-time disaster intelligence using multi-model AI")

query = st.text_input("📍 Enter Disaster Location/Event", placeholder="e.g., Florida Hurricane 2025")

if st.button("🚀 Launch AI Analysis"):
    if not query:
        st.error("Please enter a disaster name or location.")
    else:
        with st.spinner("Analyzing..."):
            data = fetch_disaster_data(query, demo_mode)
            news_texts = [f"{n['title']}: {n.get('snippet', '')}" for n in data["news"]]
            if not news_texts:
                st.warning("No news data.")
                st.stop()
            result = analyze_disaster(query, news_texts, data["geo"])
        
        # Output
        st.success("✅ Analysis Ready")
        st.markdown(f"""
        <div style='background-color:#f54242; padding:10px; border-radius:8px; color:white'>
        <h3>{result['type']}</h3>
        <h2>Severity: {result['severity']}/10</h2>
        <p>{result['severity_rationale']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Map
        if result["geo"]:
            try:
                m = folium.Map(location=[float(result["geo"]["lat"]), float(result["geo"]["lon"])], zoom_start=6)
                folium.Marker(
                    [float(result["geo"]["lat"]), float(result["geo"]["lon"])],
                    popup=result["geo"].get("display_name", "Disaster Area"),
                    icon=folium.Icon(color="red")
                ).add_to(m)
                folium_static(m)
            except:
                st.warning("Map error.")

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📅 Timeline", "🛠️ Response", "📰 News"])
        with tab1:
            for e in result["timeline"]:
                st.markdown(f"⏱️ {e}")
        with tab2:
            st.subheader("Actions")
            for a in result["actions"]:
                st.markdown(f"- {a}")
            st.subheader("Resources")
            for r in result["resources"]:
                st.markdown(f"📦 {r}")
            st.markdown(f"**Sentiment**: {result['sentiment_label']} ({result['sentiment_score']:.2f})")
        with tab3:
            for n in data["news"]:
                st.markdown(f"### {n['title']}\n{n.get('snippet', '')}\n[Read more]({n.get('link', '#')})")

# Footer
st.markdown("---")
st.markdown("Built with 🤖 using Groq, HuggingFace, SerpAPI, and Streamlit")
