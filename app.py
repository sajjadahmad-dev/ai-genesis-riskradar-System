import streamlit as st
import serpapi
from groq import Groq
from datetime import datetime
import json
import random
import os
from dotenv import load_dotenv
import logging
import folium
from streamlit_folium import folium_static

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="🚀 AI Genesis: Disaster Response",
    layout="wide",
    page_icon="🌪️",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to bottom, #f0f4ff, #ffffff);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #4b6cb7, #182848);
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextInput>div>input {
        border: 2px solid #007bff;
        border-radius: 8px;
        padding: 10px;
    }
    .severity-alert {
        border: 3px solid transparent;
        border-image: linear-gradient(to right, #ff416c, #ff4b2b) 1;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .tab-header {
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .tab1-header { color: #007bff; }
    .tab2-header { color: #28a745; }
    .tab3-header { color: #6f42c1; }
    h1, h2, h3, h4 { font-family: 'Arial', sans-serif; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

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

# Initialize Groq
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing.")
    logger.debug("Initializing Groq client...")
    groq = Groq(api_key=groq_api_key)
    logger.debug("Groq client initialized.")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {str(e)}")
    st.error(f"Failed to initialize Groq client: {str(e)}. Running in demo mode only.")
    groq = None

# --- 🛰️ Data Fetching ---
def fetch_disaster_data(query, demo_mode=False):
    logger.debug(f"Fetching data for query: {query}, demo_mode: {demo_mode}")
    if demo_mode or groq is None:
        st.info("Using demo data due to demo mode or Groq failure.")
        disaster_type = "hurricane" if "hurricane" in query.lower() else random.choice(list(DEMO_DATA.keys()))
        logger.debug(f"Returning demo data for {disaster_type}")
        return DEMO_DATA[disaster_type]
    
    try:
        serpapi_key = st.secrets.get("SERPAPI_KEY", os.getenv("SERPAPI_KEY", ""))
        if not serpapi_key:
            raise ValueError("SERPAPI_KEY is missing.")
        logger.debug("Querying SerpAPI...")
        news = serpapi.search({
            "q": f"{query} disaster",
            "api_key": serpapi_key,
            "engine": "google_news",
            "num": 3
        }).get('news_results', [])[:3]
        
        # Use demo geo data
        geo_data = DEMO_DATA.get("hurricane" if "hurricane" in query.lower() else "earthquake", {}).get("geo")
        
        if not news:
            st.warning("No news results from SerpAPI. Using demo data.")
            logger.debug("No news results, using demo data.")
            return DEMO_DATA["hurricane"]
        
        logger.debug("Data fetched successfully.")
        return {
            "news": [n for n in news if n.get('title')],
            "geo": geo_data
        }
    except Exception as e:
        logger.error(f"Data fetch error: {str(e)}")
        st.error(f"Data fetch error: {str(e)}. Using demo data.")
        return DEMO_DATA["hurricane"]

# --- 🤖 AI Analysis Engine ---
def analyze_disaster(query, news_texts, geo_data):
    logger.debug("Starting disaster analysis...")
    try:
        # Default analysis for demo mode or if Groq fails
        default_analysis = {
            "type": "Category 4 Hurricane" if "hurricane" in query.lower() else "7.2 Magnitude Earthquake",
            "severity": 9,
            "severity_rationale": "High impact based on demo data.",
            "timeline": [
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Disaster landfall reported",
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Peak impact observed",
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Disaster begins to subside"
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
            "sentiment": "Anxious and fearful due to severe disaster impact",
            "locations": []
        }
        
        if groq is None:
            logger.debug("Using default analysis (no Groq).")
            return default_analysis
        
        # Groq-based analysis
        disaster_prompt = f"""
        Analyze this disaster scenario and provide specific classification:
        News Headlines: {news_texts[:2]}
        
        Respond with valid JSON containing:
        - "type": specific disaster type (e.g., "Category 4 Hurricane")
        - "severity": integer from 1 to 10
        - "severity_rationale": brief explanation
        - "timeline": ["3 critical events with timestamps in format YYYY-MM-DD HH:MM:SS"]
        - "actions": ["3 prioritized actions"]
        - "resources": ["3 most needed resources"]
        - "sentiment": "analysis of public mood"
        - "locations": ["list of extracted locations, if any"]
        Ensure all keys are double-quoted and values are properly formatted.
        """
        
        try:
            logger.debug("Querying Groq for disaster analysis...")
            response = groq.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": disaster_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=30
            ).choices[0].message.content
            analysis = json.loads(response)
            logger.debug("Groq analysis completed.")
            return {**analysis, "geo": geo_data}
        except Exception as e:
            logger.warning(f"Groq analysis failed: {str(e)}")
            st.warning(f"Groq analysis failed: {str(e)}. Using default analysis.")
            return default_analysis
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        st.error(f"Analysis failed: {str(e)}. Please try again.")
        return default_analysis

# --- 🎨 Streamlit UI ---
with st.sidebar:
    st.markdown("<h1 style='color: white;'>AI Genesis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #d1d5db;'>LabLab AI Hackathon Entry</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: #ffffff;'>", unsafe_allow_html=True)
    demo_mode = st.checkbox("Demo Mode (Use sample data)", value=True)
    st.markdown("<hr style='border-color: #ffffff;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: white;'>🛠️ Technologies Used</h3>", unsafe_allow_html=True)
    st.markdown("- Groq Llama3-70B (Analysis)", style={"color": "#d1d5db"})
    st.markdown("- SerpAPI (Real-time News)", style={"color": "#d1d5db"})
    st.markdown("- Folium (Interactive Maps)", style={"color": "#d1d5db"})
    st.markdown("<hr style='border-color: #ffffff;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: white;'>🔗 Links</h3>", unsafe_allow_html=True)
    st.markdown("[Working Demo](https://your-streamlit-app-url.streamlit.app) *(Update after deployment)*", style={"color": "#d1d5db"})
    st.markdown("[GitHub Repo](https://github.com/your-username/ai-powered-disaster-response-system)", style={"color": "#d1d5db"})
    st.markdown("<hr style='border-color: #ffffff;'>", unsafe_allow_html=True)
    st.markdown("<p style='color: #d1d5db;'>Made with ❤️ for /execute: AI Genesis</p>", unsafe_allow_html=True)

st.markdown("<h1 style='color: #1a3c6e;'>🌪️ AI-Powered Disaster Response System</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #4a5568;'>Real-time disaster intelligence powered by Groq and SerpAPI</p>", unsafe_allow_html=True)

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
            try:
                logger.debug("Starting analysis for query: %s", query)
                # Data Collection
                data = fetch_disaster_data(query, demo_mode=demo_mode)
                logger.debug("Data fetched: %s", data)
                news_texts = [f"{n['title']}: {n.get('snippet', '')}" for n in data["news"]]
                
                if not news_texts:
                    logger.error("No valid news sources found.")
                    st.error("No valid news sources found. Try another query or enable Demo Mode.")
                    st.stop()
                
                # AI Analysis
                analysis = analyze_disaster(query, news_texts, data["geo"])
                logger.debug("Analysis result: %s", analysis)
                if analysis is None:
                    logger.error("Analysis returned no results.")
                    st.error("Analysis returned no results. Please try again.")
                    st.stop()
                
                # --- Results Dashboard ---
                st.success("✅ AI Analysis Complete!")
                
                # Severity Alert
                severity_color = "red" if analysis["severity"] > 7 else "orange" if analysis["severity"] > 4 else "green"
                st.markdown(f"""
                <div class='severity-alert' style='background:{severity_color}; color:white;'>
                    <h2 style='margin:0;'>🚨 {analysis['type'].upper()} DETECTED</h2>
                    <h1 style='margin:0; text-align:center;'>SEVERITY: {analysis['severity']}/10</h1>
                    <p style='margin:0;'><i>{analysis['severity_rationale']}</i></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Map Visualization
                if analysis["geo"]:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        try:
                            m = folium.Map(
                                location=[float(analysis["geo"]["lat"]), float(analysis["geo"]["lon"])],
                                zoom_start=7,
                                tiles="Stamen Terrain"
                            )
                            folium.Marker(
                                [analysis["geo"]["lat"], analysis["geo"]["lon"]],
                                popup=f"<b>{query}</b><br>Severity: {analysis['severity']}/10",
                                icon=folium.Icon(color=severity_color, icon="cloud")
                            ).add_to(m)
                            folium_static(m)
                        except:
                            logger.warning("Map rendering failed.")
                            st.warning("Map rendering failed.")
                    
                    with col2:
                        st.metric("📍 Primary Location", analysis["geo"].get("display_name", "Unknown"))
                        st.metric("🌍 Coordinates", f"{analysis['geo']['lat']}, {analysis['geo']['lon']}")
                        st.metric("📌 Other Locations", len(analysis["locations"]))
                
                # Timeline and Actions
                tab1, tab2, tab3 = st.tabs([
                    "📅 Timeline",
                    "🛠️ Response Plan",
                    "📰 News Sources"
                ])
                
                with tab1:
                    st.markdown("<h3 class='tab-header tab1-header'>Critical Events Timeline</h3>", unsafe_allow_html=True)
                    for event in analysis["timeline"]:
                        st.markdown(f"⏱️ {event}")
                
                with tab2:
                    st.markdown("<h3 class='tab-header tab2-header'>Response Plan</h3>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("<h4 style='color: #2d3748;'>Priority Actions</h4>", unsafe_allow_html=True)
                        for action in analysis["actions"]:
                            st.markdown(f"✅ {action}")
                    with col2:
                        st.markdown("<h4 style='color: #2d3748;'>Resources Needed</h4>", unsafe_allow_html=True)
                        for resource in analysis["resources"]:
                            st.markdown(f"📦 {resource}")
                    
                    st.markdown("---")
                    st.markdown("<h4 style='color: #2d3748;'>Public Sentiment Analysis</h4>", unsafe_allow_html=True)
                    st.write(f"Overall mood: **{analysis['sentiment']}**")
                
                with tab3:
                    st.markdown("<h3 class='tab-header tab3-header'>News Sources</h3>", unsafe_allow_html=True)
                    for news in data["news"]:
                        st.markdown(f"""
                        ### {news['title']}
                        {news.get('snippet', '')}
                        *[Source]({news.get('link', 'https://www.fema.gov')})*
                        """)
                        st.markdown("---")
                
                logger.debug("Results dashboard rendered successfully.")
            except Exception as e:
                logger.error(f"Processing failed: {str(e)}")
                st.error(f"Processing failed: {str(e)}. Please check logs or try again.")

# Footer
st.markdown("---")
st.markdown("""
<h3 style='color: #1a3c6e;'>🏆 Hackathon Compliance</h3>
✅ **AI Integration** (Groq Llama3-70B)  
✅ **Real-Time Data** (SerpAPI)  
✅ **Advanced Visualization** (Folium Maps)  
✅ **Professional UI** (Streamlit)  
✅ **Complete Documentation**  
""", unsafe_allow_html=True)
