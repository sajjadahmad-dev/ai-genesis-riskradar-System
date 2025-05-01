import streamlit as st
import serpapi
from groq import Groq
import os
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd
import folium
from streamlit_folium import folium_static

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="AI Genesis: RiskRadar", page_icon="📡")

# --- Demo Data ---
DEMO_DATA = [
    {"title": "Hurricane Disrupts Florida Supply Chains", "source": "Bloomberg", "link": "https://bloomberg.com"},
    {"title": "Insurance Costs Rise Post-Hurricane", "source": "Reuters", "link": "https://reuters.com"},
    {"title": "Florida Retail Faces Recovery Challenges", "source": "Forbes", "link": "https://forbes.com"},
    {"title": "Construction Boom Expected After Storm", "source": "CNBC", "link": "https://cnbc.com"},
    {"title": "Energy Sector Braces for Hurricane Impact", "source": "TechCrunch", "link": "https://techcrunch.com"}
]

# Initialize Groq
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing.")
    groq = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Failed to connect to Groq: {str(e)}. Using demo mode.")
    groq = None

# --- Fetch News ---
def fetch_news(query, demo_mode=False):
    if demo_mode or groq is None:
        st.info("Showing sample news (demo mode).")
        return DEMO_DATA

    try:
        serpapi_key = st.secrets.get("SERPAPI_KEY", os.getenv("SERPAPI_KEY", ""))
        if not serpapi_key:
            raise ValueError("SERPAPI_KEY is missing.")
        results = serpapi.search({
            "q": f"{query} business impact",
            "api_key": serpapi_key,
            "engine": "google",
            "num": 5
        }).get('organic_results', [])[:5]

        if not results:
            st.warning("No news found. Showing sample news.")
            return DEMO_DATA

        return [
            {
                "title": r.get("title", ""),
                "source": r.get("source", "Unknown"),
                "link": r.get("link", "#")
            }
            for r in results if r.get("title")
        ]
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}. Showing sample news.")
        return DEMO_DATA

# --- Analyze Risks/Opportunities ---
def analyze_news(articles, query):
    if groq is None:
        return "Contact news sources for risk and opportunity insights."

    try:
        titles = "; ".join([a['title'] for a in articles])
        prompt = f"""
        Analyze these news titles: "{titles}" for the topic "{query}".
        Focus on business risks, market threats, or growth opportunities, including disaster-related impacts.
        Provide a 2-3 sentence summary and a short strategy for a startup or investor.
        """
        response = groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=10
        ).choices[0].message.content.strip()
        return response
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}. Using default response.")
        return "Contact news sources for risk and opportunity insights."

# --- Risk Location Map ---
def show_risk_map(query):
    # Default coordinates (Florida)
    location = {
        "Hurricane Florida": [27.994402, -81.760254],
        "Earthquake California": [36.778259, -119.417931],
        "Flood Pakistan": [30.3753, 69.3451],
        "Typhoon Japan": [36.2048, 138.2529],
        "Cyclone India": [20.5937, 78.9629]
    }

    # Find coordinates for known queries, fallback to center of USA
    coords = location.get(query.strip(), [37.0902, -95.7129])
    
    m = folium.Map(location=coords, zoom_start=6)
    folium.Marker(
        location=coords,
        popup=f"Risk Zone: {query}",
        tooltip="Click for location",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    st.header("📍 Risk Map")
    folium_static(m)

# --- Streamlit UI ---
st.title("📡 AI Genesis: RiskRadar")
st.write("Analyze business risks and opportunities from recent news.")

with st.sidebar:
    st.header("AI Genesis: RiskRadar")
    st.write("🔬 Built for LabLab AI Hackathon: /execute: AI Genesis")
    demo_mode = st.checkbox("Demo Mode", value=True)
    st.header("🛠 Technologies")
    st.write("- Groq LLaMA3-70B")
    st.write("- SerpAPI")
    st.write("- Plotly")
    st.write("- Folium")
    st.header("🔗 Links")
    st.write("[🚀 Demo](https://your-streamlit-app-url.streamlit.app)")
    st.write("[💻 GitHub](https://github.com/your-username/riskradar-ai)")

query = st.text_input("🔍 Enter a business topic:", placeholder="e.g., Hurricane Florida")

if st.button("🔎 Analyze"):
    if not query:
        st.error("Please enter a business topic.")
    else:
        with st.spinner("Fetching news and analyzing..."):
            articles = fetch_news(query, demo_mode=demo_mode)

            if not articles:
                st.error("No news found. Try another topic.")
                st.stop()

            analysis = analyze_news(articles, query)

            st.success("✅ Analysis Complete!")

            # Show map
            show_risk_map(query)

            # Bar chart of sources
            source_counts = pd.Series([a['source'] for a in articles]).value_counts().reset_index()
            source_counts.columns = ['Source', 'Count']
            fig = px.bar(source_counts, x='Source', y='Count', title="News Sources", color='Source')
            st.plotly_chart(fig)

            # Analysis summary
            st.header("📈 Strategic Insights")
            st.write(analysis)

            # News table
            st.header("📰 Related News")
            df = pd.DataFrame(articles)
            st.dataframe(df[["title", "source", "link"]], use_container_width=True)
