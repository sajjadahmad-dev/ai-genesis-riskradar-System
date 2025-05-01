import streamlit as st
import serpapi
from groq import Groq
import os
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="AI Genesis: RiskRadar", page_icon="📡")

# --- Demo Data ---
DEMO_DATA = [
    {"title": "AI Healthcare Startup Raises $100M", "source": "TechCrunch", "link": "https://techcrunch.com"},
    {"title": "Regulatory Challenges for AI in Medicine", "source": "Reuters", "link": "https://reuters.com"},
    {"title": "New AI Diagnostic Tool Approved", "source": "Bloomberg", "link": "https://bloomberg.com"},
    {"title": "Data Privacy Concerns in AI Health", "source": "Forbes", "link": "https://forbes.com"},
    {"title": "AI Health Market to Grow 20%", "source": "CNBC", "link": "https://cnbc.com"}
]

# Initialize Groq
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing.")
    groq = Groq(api_key=groq_api_key)
except Exception:
    st.error("Cannot connect to Groq. Using demo mode.")
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
            "q": query,
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
    except Exception:
        st.error("Error fetching news. Showing sample news.")
        return DEMO_DATA

# --- Analyze Risks/Opportunities ---
def analyze_news(articles, query):
    if groq is None:
        return "Contact news sources for risk and opportunity insights."
    
    try:
        titles = "; ".join([a['title'] for a in articles])
        prompt = f"""
        Analyze these news titles: "{titles}" for the topic "{query}".
        Summarize business risks, market threats, or growth opportunities in 2-3 sentences.
        Provide a short strategy for a startup or investor.
        """
        response = groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=10
        ).choices[0].message.content.strip()
        return response
    except Exception:
        return "Contact news sources for risk and opportunity insights."

# --- Streamlit UI ---
st.title("AI Genesis: RiskRadar")
st.write("Analyze business risks and opportunities from recent news.")

with st.sidebar:
    st.header("AI Genesis: RiskRadar")
    st.write("LabLab AI Hackathon")
    demo_mode = st.checkbox("Demo Mode", value=True)
    st.header("Technologies")
    st.write("- Groq Llama3-70B")
    st.write("- SerpAPI")
    st.write("- Plotly")
    st.header("Links")
    st.write("[Demo](https://your-streamlit-app-url.streamlit.app)")
    st.write("[GitHub](https://github.com/your-username/ai-powered-disaster-response-system)")
    st.write("Built for /execute: AI Genesis")

query = st.text_input("Enter a business topic:", placeholder="e.g., AI in Healthcare")

if st.button("Analyze"):
    if not query:
        st.error("Please enter a business topic.")
    else:
        with st.spinner("Fetching news and analyzing..."):
            # Fetch news
            articles = fetch_news(query, demo_mode=demo_mode)
            
            if not articles:
                st.error("No news found. Try another topic.")
                st.stop()
            
            # Analyze news
            analysis = analyze_news(articles, query)
            
            # --- Results ---
            st.success("Analysis Complete!")
            
            # News Source Bar Chart
            source_counts = pd.Series([a['source'] for a in articles]).value_counts().reset_index()
            source_counts.columns = ['Source', 'Count']
            fig = px.bar(
                source_counts,
                x='Source',
                y='Count',
                title="News Sources",
                color='Source'
            )
            st.plotly_chart(fig)
            
            # Analysis
            st.header("Strategic Insights")
            st.write(analysis)
            
            # News List
            st.header("Related News")
            df = pd.DataFrame(articles)
            st.dataframe(df[["title", "source", "link"]], use_container_width=True)
