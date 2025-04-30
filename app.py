import streamlit as st
import serpapi
import requests
from groq import Groq
from transformers import pipeline
import folium
from streamlit_folium import folium_static
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

# --- 🔐 SECRETS MANAGEMENT ---
def get_secrets():
    """Handle secrets with multiple fallback options"""
    secrets = {
        "GROQ_API_KEY": None,
        "SERPAPI_KEY": None
    }
    
    # 1. Try Streamlit secrets first
    try:
        if hasattr(st, 'secrets'):
            secrets["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY")
            secrets["SERPAPI_KEY"] = st.secrets.get("SERPAPI_KEY")
    except Exception as e:
        st.error(f"Secrets access error: {str(e)}")
    
    # 2. Try environment variables
    if not all(secrets.values()):
        try:
            import os
            secrets["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")
            secrets["SERPAPI_KEY"] = os.environ.get("SERPAPI_KEY")
        except Exception as e:
            st.error(f"Env vars error: {str(e)}")
    
    # 3. Final fallback (remove before deployment)
    if not all(secrets.values()):
        secrets["GROQ_API_KEY"] = "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"
        secrets["SERPAPI_KEY"] = "39a147d2d97b7b81d98fe00e15a15edfa4e701f465c2f46df26ed534ef2cbd50"
    
    return secrets

secrets = get_secrets()

# --- 🚀 INITIALIZE SERVICES ---
try:
    groq = Groq(api_key=secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"Failed to initialize Groq client: {str(e)}")
    st.stop()

@st.cache_resource
def load_ai_models():
    try:
        return {
            "ner": pipeline("ner", model="dslim/bert-base-NER")
        }
    except Exception as e:
        st.error(f"Failed to load AI models: {str(e)}")
        return None

models = load_ai_models()
if models is None:
    st.stop()

# --- 🛰️ DATA FUNCTIONS ---
def fetch_disaster_data(query, demo_mode=False):
    if demo_mode:
        return DEMO_DATA["hurricane"]
    
    try:
        news = serpapi.search({
            "q": f"{query} disaster",
            "api_key": secrets["SERPAPI_KEY"],
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

# --- 🤖 AI ANALYSIS ---
def analyze_disaster(query, news_texts, geo_data):
    try:
        # Entity Recognition
        entities = models["ner"](" ".join(news_texts))
        locations = list({e['word'] for e in entities if e['entity'] in ['B-LOC','I-LOC']})
        
        # Groq Analysis
        prompt = f"""
        Analyze this disaster scenario:
        Query: {query}
        News: {news_texts[:2]}
        
        Return JSON with:
        - "type": specific disaster classification
        - "severity": 1-10 scale
        - "actions": ["3 response actions"]
        - "resources": ["3 needed resources"]
        """
        
        response = groq.chat.completions.create(
            model="llama3-8b-8192",
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

# --- 🎨 STREAMLIT UI ---
st.set_page_config(
    page_title="AI Disaster Response",
    layout="centered",
    page_icon="🌪️"
)

# Sidebar
with st.sidebar:
    st.title("Settings")
    demo_mode = st.checkbox("Demo Mode", value=True)
    st.markdown("---")
    st.markdown("**Models Used:**")
    st.markdown("- BERT-NER (Location Extraction)")
    st.markdown("- Llama3-8B (Analysis)")

# Main Interface
st.title("🌪️ AI Disaster Response System")
st.markdown("Real-time disaster analysis powered by AI")

query = st.text_input("Enter disaster location:", "Florida Hurricane")

if st.button("Analyze", type="primary"):
    with st.spinner("Processing..."):
        data = fetch_disaster_data(query, demo_mode)
        if not data or not data["news"]:
            st.error("No data available for this location")
            st.stop()
            
        news_texts = [f"{n['title']}: {n.get('snippet', '')}" for n in data["news"]]
        analysis = analyze_disaster(query, news_texts, data["geo"])
        
        if not analysis:
            st.stop()
        
        # Display Results
        st.subheader(f"🚨 {analysis['type'].upper()} DETECTED")
        st.progress(analysis['severity']/10, text=f"Severity: {analysis['severity']}/10")
        
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
            folium_static(m, width=700)
        
        # Action Plan
        st.subheader("🛠️ Response Plan")
        for action in analysis.get("actions", []):
            st.markdown(f"- {action}")
        
        # Resources
        st.subheader("📦 Needed Resources")
        for resource in analysis.get("resources", []):
            st.markdown(f"- {resource}")

# Footer
st.markdown("---")
st.caption("Built for AI Genesis Hackathon | Remove API keys before deployment")
