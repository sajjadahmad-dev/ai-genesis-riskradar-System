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
st.set_page_config(page_title="🛡️ AI Genesis: Aid Finder", page_icon="🛡️", layout="wide")

# --- Custom CSS for beautiful styling ---
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
    }
    h1 {
        color: #007bff;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: scale(1.05);
    }
    .stSidebar {
        background-color: #f1f3f5;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Demo Data ---
DEMO_DATA = {
    "hurricane": [
        {"name": "Miami Red Cross", "description": "Offers shelter and meals.", "link": "https://www.redcross.org", "type": "Shelter"},
        {"name": "Florida Food Bank", "description": "Provides free food supplies.", "link": "https://www.feedingamerica.org", "type": "Food Bank"},
        {"name": "FEMA Miami", "description": "Coordinates disaster aid.", "link": "https://www.fema.gov", "type": "Aid Agency"}
    ],
    "earthquake": [
        {"name": "Tokyo Shelter Network", "description": "Temporary housing for survivors.", "link": "#", "type": "Shelter"},
        {"name": "Japan Food Aid", "description": "Distributes meals.", "link": "#", "type": "Food Bank"},
        {"name": "Global Relief NGO", "description": "International aid support.", "link": "#", "type": "Aid Agency"}
    ]
}

# --- Groq Init ---
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing.")
    groq = Groq(api_key=groq_api_key)
except Exception:
    st.error("Cannot connect to Groq. Using demo mode.")
    groq = None

# --- Fetch Resources ---
def fetch_resources(query, demo_mode=False):
    if demo_mode or groq is None:
        st.info("💡 Showing sample resources (demo mode).")
        disaster_type = "hurricane" if "hurricane" in query.lower() else "earthquake"
        return DEMO_DATA[disaster_type]
    
    try:
        serpapi_key = st.secrets.get("SERPAPI_KEY", os.getenv("SERPAPI_KEY", ""))
        if not serpapi_key:
            raise ValueError("SERPAPI_KEY is missing.")
        results = serpapi.search({
            "q": f"{query} disaster relief resources",
            "api_key": serpapi_key,
            "engine": "google",
            "num": 3
        }).get('organic_results', [])[:3]
        
        if not results:
            st.warning("⚠️ No resources found. Showing sample resources.")
            return DEMO_DATA["hurricane"]
        
        types = ["Shelter", "Food Bank", "Aid Agency"]
        return [
            {
                "name": r.get("title", ""),
                "description": r.get("snippet", ""),
                "link": r.get("link", "#"),
                "type": types[i % len(types)]
            }
            for i, r in enumerate(results) if r.get("title")
        ]
    except Exception:
        st.error("❌ Error fetching resources. Showing sample resources.")
        return DEMO_DATA["hurricane"]

# --- Generate Aid Tips ---
def generate_aid_tips(query, resources):
    tips = []
    
    if groq is None:
        return [f"Contact {r['name']} for assistance." for r in resources]
    
    try:
        for resource in resources:
            prompt = f"""
            For the disaster '{query}' and resource '{resource['name']}' ({resource['type']}), provide a short aid tip (1 sentence).
            """
            try:
                response = groq.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    timeout=10
                ).choices[0].message.content.strip()
                tips.append(response)
            except Exception:
                tips.append(f"Contact {resource['name']} for assistance.")
    except Exception:
        tips = [f"Contact {r['name']} for assistance." for r in resources]
    
    return tips

# --- Streamlit UI ---
st.title("🛡️ AI Genesis: Aid Finder")
st.subheader("🌍 Locate disaster relief resources and get instant AI-powered aid tips")

with st.sidebar:
    st.header("⚙️ Settings")
    demo_mode = st.checkbox("Demo Mode", value=True)
    
    st.markdown("---")
    st.header("🧠 Tech Stack")
    st.markdown("- Groq `LLaMA3-70B`\n- SerpAPI\n- Plotly\n- Streamlit\n- OpenAI")
    
    st.markdown("---")
    st.header("🔗 Quick Links")
    st.markdown("[🌐 Live Demo](https://your-streamlit-app-url.streamlit.app)")
    st.markdown("[💻 GitHub](https://github.com/your-username/ai-powered-disaster-response-system)")
    st.caption("🏆 Built for /execute: AI Genesis Hackathon")

query = st.text_input("📍 Enter Location and Disaster:", placeholder="e.g., Miami Hurricane")

if st.button("🔍 Find Resources"):
    if not query:
        st.error("❗ Please enter a location and disaster.")
    else:
        with st.spinner("🔎 Fetching resources..."):
            resources = fetch_resources(query, demo_mode=demo_mode)
            
            if not resources:
                st.error("No resources found. Try another query.")
                st.stop()
            
            tips = generate_aid_tips(query, resources)
            
            st.success("✅ Resources Found!")

            # Pie Chart of Resource Types
            type_counts = pd.Series([r['type'] for r in resources]).value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']
            fig = px.pie(
                type_counts,
                names='Type',
                values='Count',
                title="🧩 Resource Types",
                color='Type',
                color_discrete_map={'Shelter': '#28a745', 'Food Bank': '#007bff', 'Aid Agency': '#6c757d'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Resource Cards
            st.header("📦 Relief Resources")
            for resource, tip in zip(resources, tips):
                st.markdown(f"""
                <div style='padding:1rem; margin-bottom:1rem; border-left:5px solid #007bff; background-color:#f1f3f5; border-radius:8px'>
                    <h4 style='margin-bottom:0.2rem;'>{resource['name']}</h4>
                    <p style='margin:0'><b>Type:</b> {resource['type']}</p>
                    <p style='margin:0'><b>Description:</b> {resource.get('description', '')}</p>
                    <p style='margin:0'><b>💡 Tip:</b> {tip}</p>
                    <a href='{resource.get('link', '#')}' target='_blank'>🔗 Visit Resource</a>
                </div>
                """, unsafe_allow_html=True)
