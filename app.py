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

# Set page config as the FIRST Streamlit command
st.set_page_config(
    page_title="🚀 AI Genesis: Disaster Response",
    layout="wide",
    page_icon="🌪️",
    initial_sidebar_state="expanded"
)

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

# --- 🧠 AI Model Initialization ---
@st.cache_resource
def load_ai_models():
    return {
        "disaster_clf": pipeline("text-classification", model="distilbert-base-uncased"),
        "ner": pipeline("ner", model="dslim/bert-base-NER"),
        "sentiment": pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
    }

models = load_ai_models()

# Fetch API keys from Streamlit secrets
groq = Groq(api_key=st.secrets["API_GROQ"])
serpapi_key = st.secrets["SERP_API_KEY"]

# --- 🛰️ Data Fetching ---
def fetch_disaster_data(query, demo_mode=False):
    if demo_mode:
        disaster_type = random.choice(list(DEMO_DATA.keys()))
        return DEMO_DATA[disaster_type]
    
    try:
        news = serpapi.search({
            "q": f"{query} disaster",
            "api_key": serpapi_key,
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
    # Entity Recognition
    try:
        entities = models["ner"](" ".join(news_texts))
        locations = list({e['word'] for e in entities if e['entity'].startswith('B-LOC') or e['entity'].startswith('I-LOC')})
    except Exception as e:
        locations = []
        st.warning(f"Location extraction failed: {str(e)}")
    
    # Disaster Classification with Groq
    disaster_prompt = f"""
    Analyze this disaster scenario and provide specific classification:
    News Headlines: {news_texts[:2]}
    
    Respond with valid JSON containing:
    - "type": specific disaster type (e.g., "Category 4 Hurricane")
    - "severity": integer from 1 to 10 (e.g., 9, not "9/10")
    - "severity_rationale": brief explanation
    Ensure all keys are double-quoted and values are properly formatted (e.g., severity as a number).
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
    
    # Generate Response Plan
    response_prompt = f"""
    Generate a detailed response plan for:
    Disaster: {disaster_analysis['type']}
    Severity: {disaster_analysis['severity']}/10
    Locations: {locations or 'None'}
    
    Provide valid JSON with:
    - "timeline": ["3 critical events with timestamps in format YYYY-MM-DD HH:MM:SS"]
    - "actions": ["3 prioritized actions"]
    - "resources": ["3 most needed resources"]
    - "sentiment": "analysis of public mood"
    Ensure all keys are double-quoted and values are properly formatted.
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
                "Evacuate high-risk areas, especially coastal and flood-prone zones.",
                "Activate emergency services, including responders and rescue teams.",
                "Establish communication networks for affected areas."
            ],
            "resources": [
                "Food and water (100,000 units)",
                "Medical supplies (50,000 units)",
                "Generators and fuel (500 units)"
            ],
            "sentiment": "Anxious and fearful due to severe disaster impact"
        }
    
    # Sentiment Analysis
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
# Sidebar Configuration
with st.sidebar:
    st.image("https://www.google.com/url?sa=i&url=https%3A%2F%2Fmedium.com%2F%40alex_tolson%2Fthe-role-of-satellite-imagery-in-disaster-management-22255e1663a3&psig=AOvVaw0xZeb2ucoTXS47jc0NpcLc&ust=1746097443086000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCOi5voLO_4wDFQAAAAAdAAAAABAE", width=100)
    st.title("AI Genesis")
    st.markdown("**LabLab AI Hackathon Entry**")
    st.markdown("---")
    demo_mode = st.checkbox("Demo Mode (Use sample data)", value=True)
    st.markdown("---")
    st.markdown("### 🛠️ Models Used")
    st.markdown("- DistilBERT (Classification)")
    st.markdown("- BERT-NER (Location Extraction)")
    st.markdown("- Llama3-70B (Analysis)")
    st.markdown("---")
    st.markdown("Made with ❤️ for /execute: AI Genesis")

# Main Interface
st.title("🌪️ AI-Powered Disaster Response System")
st.markdown("Real-time disaster intelligence with multi-model AI analysis")

# Input Section
query = st.text_input(
    "📍 Enter Disaster Location/Event:", 
    placeholder="e.g., Florida Hurricane 2025",
    help="Enter a location or specific disaster event"
)

if st.button("🚀 Launch AI Analysis", type="primary"):
    if not query:
        st.error("Please enter a disaster query.")
    else:
        with st.spinner("🛰️ Gathering real-time intelligence..."):
            # Data Collection
            data = fetch_disaster_data(query, demo_mode=demo_mode)
            news_texts = [f"{n['title']}: {n.get('snippet', '')}" for n in data["news"]]
            
            if not news_texts:
                st.warning("No disaster-related news found.")
            else:
                # AI Analysis
                results = analyze_disaster(query, news_texts, data.get("geo"))
                
                # Display the Results
                st.header("🔍 Disaster Analysis")
                st.json(results)
                
                # Map View
                if results.get("geo"):
                    st.header("📍 Geographic Info")
                    map_obj = folium.Map(
                        location=[float(results["geo"]["lat"]), float(results["geo"]["lon"])],
                        zoom_start=6
                    )
                    folium.Marker(
                        location=[float(results["geo"]["lat"]), float(results["geo"]["lon"])],
                        popup=f"Disaster Location: {results['geo']['display_name']}"
                    ).add_to(map_obj)
                    folium_static(map_obj)
