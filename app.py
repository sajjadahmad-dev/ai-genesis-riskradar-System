import streamlit as st
import serpapi
from groq import Groq
import os
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd
import folium
from streamlit_folium import st_folium

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="AI Genesis: RiskRadar", page_icon="🛁")

# --- Demo Data ---
DEMO_DATA = [
    {"title": "Hurricane Disrupts Florida Supply Chains", "source": "Bloomberg", "link": "https://bloomberg.com", "snippet": "Major supply chain disruptions following Hurricane in Florida.", "date": "2024-09-12"},
    {"title": "Insurance Costs Rise Post-Hurricane", "source": "Reuters", "link": "https://reuters.com", "snippet": "Property insurance spikes after disaster hits East Coast.", "date": "2024-09-13"},
    {"title": "Florida Retail Faces Recovery Challenges", "source": "Forbes", "link": "https://forbes.com", "snippet": "Retail sector in Florida struggles to reopen post hurricane.", "date": "2024-09-14"},
    {"title": "Construction Boom Expected After Storm", "source": "CNBC", "link": "https://cnbc.com", "snippet": "Florida set for a surge in reconstruction contracts.", "date": "2024-09-14"},
    {"title": "Energy Sector Braces for Hurricane Impact", "source": "TechCrunch", "link": "https://techcrunch.com", "snippet": "Energy firms prepare for losses as hurricane nears Florida.", "date": "2024-09-13"}
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

        return [
            {
                "title": r.get("title", ""),
                "source": r.get("source", "Unknown"),
                "link": r.get("link", "#"),
                "snippet": r.get("snippet", "No summary available."),
                "date": r.get("date", "Unknown")
            }
            for r in results if r.get("title")
        ]
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}. Showing sample news.")
        return DEMO_DATA

# --- Analyze Risks/Opportunities ---
def analyze_news(articles, query):
    if groq is None:
        return "Unable to analyze without Groq API."

    try:
        news_input = "\n".join([f"{a['title']} - {a['snippet']}" for a in articles])
        prompt = f"""
        You are a business analyst assistant. Analyze the following news:
        {news_input}

        For the topic: {query}

        Provide:
        1. A short summary of risks and opportunities
        2. What industries or sectors are impacted?
        3. Investor strategy recommendation
        4. Classify each article as [Opportunity, Risk, Neutral]
        """

        response = groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=15
        ).choices[0].message.content.strip()

        return response
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}. Using default response.")
        return "Unable to analyze the data. Please try later."

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
    st.write("- Plotly & Folium")
    st.header("Links")
    st.write("[Demo](https://your-streamlit-app-url.streamlit.app)")
    st.write("[GitHub](https://github.com/your-username/ai-powered-disaster-response-system)")
    st.write("Built for /execute: AI Genesis")

query = st.text_input("Enter a business topic:", placeholder="e.g., Hurricane Florida Business Impact")

if st.button("Analyze"):
    if not query:
        st.error("Please enter a business topic.")
    else:
        with st.spinner("Fetching news and analyzing..."):
            articles = fetch_news(query, demo_mode=demo_mode)

            if not articles:
                st.error("No news found. Try another topic.")
                st.stop()

            analysis = analyze_news(articles, query)

            st.success("Analysis Complete!")

            # --- Chart: Source Distribution ---
            source_counts = pd.Series([a['source'] for a in articles]).value_counts().reset_index()
            source_counts.columns = ['Source', 'Count']
            fig = px.bar(source_counts, x='Source', y='Count', title="News Sources", color='Source')
            st.plotly_chart(fig)

            # --- Table: News with Metadata ---
            st.header("Related News")
            df = pd.DataFrame(articles)
            st.dataframe(df[["title", "source", "snippet", "date", "link"]], use_container_width=True)

            # --- Strategic Insights ---
            st.header("Strategic Insights")
            st.write(analysis)

            # --- Optional: Display a simple map ---
            if "florida" in query.lower():
                st.subheader("Affected Area")
                map_ = folium.Map(location=[27.994402, -81.760254], zoom_start=6)
                folium.Marker([27.994402, -81.760254], tooltip="Florida Impact Zone").add_to(map_)
                st_folium(map_, width=700, height=400)
