import streamlit as st
import serpapi
import os
import io
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

# Set Streamlit page config
st.set_page_config(
    page_title="🚀 AI Genesis: Disaster Response",
    layout="wide",
    page_icon="🌪️",
    initial_sidebar_state="expanded"
)

# --- 🧠 AI Model Initialization ---
@st.cache_resource

def load_ai_models():
    try:
        return {
            "disaster_clf": pipeline("text-classification", model="distilbert-base-uncased", local_files_only=True),
            "ner": pipeline("ner", model="dslim/bert-base-NER", local_files_only=True),
            "sentiment": pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis", local_files_only=True)
        }
    except Exception as e:
        st.error(f"Model loading failed. Make sure models are available locally. Error: {str(e)}")
        return {}

models = load_ai_models()
groq = Groq(api_key=st.secrets.get("API_GROQ"))

# --- 🛰️ Data Fetching ---
def fetch_disaster_data(query, demo_mode=False):
    if demo_mode:
        disaster_type = random.choice(list(DEMO_DATA.keys()))
        return DEMO_DATA[disaster_type]

    try:
        news = serpapi.search({
            "q": f"{query} disaster",
            "api_key": st.secrets.get("API_SERP"),
            "engine": "google_news",
            "num": 3
        }).get('news_results', [])[:3]

        geo_data = requests.get(
            f"https://nominatim.openstreetmap.org/search?q={query}&format=json",
            headers={"User-Agent": "AI-Genesis-Hackathon"}
        ).json()

        if not news:
            raise ValueError("No news results returned from SerpAPI.")

        return {
            "news": [n for n in news if n.get('title')],
            "geo": geo_data[0] if geo_data else None
        }
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}. Using demo data.")
        return DEMO_DATA["hurricane"]

# --- 🤖 AI Analysis Engine ---
def analyze_disaster(query, news_texts, geo_data):
    try:
        entities = models["ner"](" ".join(news_texts))
        locations = list({e['word'] for e in entities if e['entity'].startswith('B-LOC') or e['entity'].startswith('I-LOC')})
    except Exception as e:
        locations = []
        st.warning(f"Location extraction failed: {str(e)}")

    disaster_prompt = f"""
    Analyze this disaster scenario and provide specific classification:
    News Headlines: {news_texts[:2]}

    Respond with valid JSON containing:
    - \"type\": specific disaster type (e.g., \"Category 4 Hurricane\")
    - \"severity\": integer from 1 to 10
    - \"severity_rationale\": brief explanation
    """

    try:
        disaster_analysis = groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": disaster_prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        ).choices[0].message.content
        disaster_analysis = json.loads(disaster_analysis)
    except Exception as e:
        st.warning(f"Groq disaster analysis failed: {str(e)}. Using default analysis.")
        disaster_analysis = {
            "type": "Category 4 Hurricane" if "hurricane" in query.lower() else "Unknown Disaster",
            "severity": 9,
            "severity_rationale": "High impact based on news reports of significant damage."
        }

    response_prompt = f"""
    Generate a detailed response plan for:
    Disaster: {disaster_analysis['type']}
    Severity: {disaster_analysis['severity']}/10
    Locations: {locations or 'None'}

    Provide valid JSON with:
    - \"timeline\": ["3 critical events with timestamps in format YYYY-MM-DD HH:MM:SS"]
    - \"actions\": ["3 prioritized actions"]
    - \"resources\": ["3 most needed resources"]
    - \"sentiment\": "analysis of public mood"
    """

    try:
        response_plan = groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": response_prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        ).choices[0].message.content
        response_plan = json.loads(response_plan)
    except Exception as e:
        st.error(f"Groq response plan failed: {str(e)}. Using default response plan.")
        response_plan = {
            "timeline": [
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Hurricane landfall reported",
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Peak storm surge observed",
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Storm begins to subside"
            ],
            "actions": [
                "Evacuate high-risk areas",
                "Activate emergency services",
                "Establish communication networks"
            ],
            "resources": [
                "Food and water",
                "Medical supplies",
                "Generators and fuel"
            ],
            "sentiment": "Anxious and fearful"
        }

    try:
        sentiment = models["sentiment"](" ".join(news_texts[:3]))
        sentiment_label = sentiment[0]["label"]
        sentiment_score = sentiment[0]["score"]
    except:
        sentiment_label, sentiment_score = "Neutral", 0.5
        st.warning("Sentiment analysis failed.")

    return {
        **disaster_analysis,
        **response_plan,
        "locations": locations,
        "geo": geo_data,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score
    }

# --- 🎨 Streamlit UI ---
with st.sidebar:
    st.title("AI Genesis")
    demo_mode = st.checkbox("Demo Mode (Use sample data)", value=True)
    st.markdown("- DistilBERT (Classification)\n- BERT-NER (Location Extraction)\n- Llama3-70B (Analysis)")

st.title("🌪️ AI-Powered Disaster Response System")
st.markdown("Real-time disaster intelligence with multi-model AI analysis")

query = st.text_input("📍 Enter Disaster Location/Event:", placeholder="e.g., Florida Hurricane 2025")
if st.button("🚀 Launch AI Analysis"):
    if not query:
        st.error("Please enter a disaster query.")
    else:
        with st.spinner("🛰️ Gathering intelligence..."):
            data = fetch_disaster_data(query, demo_mode=demo_mode)
            news_texts = [f"{n['title']}: {n.get('snippet', '')}" for n in data["news"]]
            analysis = analyze_disaster(query, news_texts, data["geo"])

            severity_color = "red" if analysis["severity"] > 7 else "orange" if analysis["severity"] > 4 else "green"
            st.markdown(f"""
            <div style='background:{severity_color};padding:15px;border-radius:10px;color:white;'>
            <h2>{analysis['type']}</h2><h3>Severity: {analysis['severity']}/10</h3><p>{analysis['severity_rationale']}</p>
            </div>""", unsafe_allow_html=True)

            if analysis["geo"]:
                m = folium.Map(location=[float(analysis["geo"]["lat"]), float(analysis["geo"]["lon"])], zoom_start=7)
                folium.Marker([float(analysis["geo"]["lat"]), float(analysis["geo"]["lon"])], popup=query).add_to(m)
                folium_static(m)

            st.subheader("📌 Critical Events Timeline")
            for event in analysis["timeline"]:
                st.write("-", event)

            st.subheader("📦 Priority Actions")
            for act in analysis["actions"]:
                st.write("-", act)

            st.subheader("🚚 Required Resources")
            for res in analysis["resources"]:
                st.write("-", res)

            st.subheader("🧠 Public Sentiment")
            st.info(f"{analysis['sentiment']} (Model: {analysis['sentiment_label']} with confidence {round(analysis['sentiment_score'], 2)})")
