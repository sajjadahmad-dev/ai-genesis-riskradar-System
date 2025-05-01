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
st.set_page_config(
    page_title="AI Genesis: Aid Finder",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional styling
st.markdown("""
<style>
    .stApp {
        background: #ffffff;
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
        border-radius: 5px;
        padding: 8px 16px;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextInput>div>input {
        border: 1px solid #007bff;
        border-radius: 5px;
        padding: 8px;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    h1, h2, h3, p {
        font-family: Arial, sans-serif;
        color: #1a3c6e;
    }
    .stMarkdown p {
        color: #333333;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# --- 🌟 Demo Data ---
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

# Initialize Groq
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing.")
    groq = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Cannot connect to Groq: {str(e)}. Using demo mode.")
    groq = None

# --- 🛠️ Fetch Resources ---
def fetch_resources(query, demo_mode=False):
    if demo_mode or groq is None:
        st.info("Showing sample resources (demo mode).")
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
            st.warning("No resources found. Showing sample resources.")
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
    except Exception as e:
        st.error(f"Error fetching resources: {str(e)}. Showing sample resources.")
        return DEMO_DATA["hurricane"]

# --- 🤖 Generate Aid Tips ---
def generate_aid_tips(query, resources):
    tips = []
    
    if groq is None:
        return [f"Contact {r['name']} for assistance." for r in resources]
    
    try:
        for resource in resources:
            prompt = f"""
            You are a disaster relief assistant. For the disaster '{query}' and resource '{resource['name']}' ({resource['type']}), provide a short aid tip (1 sentence). Example: "Contact Miami Red Cross for safe shelter and meals."
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

# --- 🎨 Streamlit UI ---
with st.sidebar:
    st.markdown("<h1 style='color: white;'>AI Genesis: Aid Finder</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #d1d5db; font-size: 14px;'>LabLab AI Hackathon</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: #ffffff;'>", unsafe_allow_html=True)
    demo_mode = st.checkbox("Demo Mode", value=True)
    st.markdown("<hr style='border-color: #ffffff;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: white;'>Technologies</h3>", unsafe_allow_html=True)
    st.markdown("- Groq Llama3-70B")
    st.markdown("- SerpAPI")
    st.markdown("- Plotly")
    st.markdown("<hr style='border-color: #ffffff;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: white;'>Links</h3>", unsafe_allow_html=True)
    st.markdown("[Demo](https://your-streamlit-app-url.streamlit.app)")
    st.markdown("[GitHub](https://github.com/your-username/ai-powered-disaster-response-system)")
    st.markdown("<hr style='border-color: #ffffff;'>", unsafe_allow_html=True)
    st.markdown("<p style='color: #d1d5db; font-size: 14px;'>Built for /execute: AI Genesis</p>", unsafe_allow_html=True)

st.markdown("<h1>AI Genesis: Aid Finder</h1>", unsafe_allow_html=True)
st.markdown("<p>Locate disaster relief resources and get practical aid tips.</p>", unsafe_allow_html=True)

query = st.text_input(
    "Enter Location and Disaster:", 
    placeholder="e.g., Miami Hurricane",
    help="Type a location and disaster type"
)

if st.button("Find Resources"):
    if not query:
        st.error("Please enter a location and disaster.")
    else:
        with st.spinner("Fetching resources..."):
            # Fetch resources
            resources = fetch_resources(query, demo_mode=demo_mode)
            
            if not resources:
                st.error("No resources found. Try another query.")
                st.stop()
            
            # Generate aid tips
            tips = generate_aid_tips(query, resources)
            
            # --- Results ---
            st.success("Resources Found!")
            
            # Resource Type Pie Chart
            type_counts = pd.Series([r['type'] for r in resources]).value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']
            fig = px.pie(
                type_counts,
                names='Type',
                values='Count',
                title="Resource Types",
                color='Type',
                color_discrete_map={'Shelter': '#28a745', 'Food Bank': '#007bff', 'Aid Agency': '#6c757d'}
            )
            fig.update_layout(margin=dict(t=50, b=50, l=50, r=50))
            st.plotly_chart(fig, use_container_width=True)
            
            # Resource List
            st.markdown("<h3>Relief Resources</h3>", unsafe_allow_html=True)
            for resource, tip in zip(resources, tips):
                st.markdown(f"""
                **{resource['name']}**  
                *Type*: {resource['type']}  
                *Description*: {resource.get('description', '')}  
                *Aid Tip*: {tip}  
                [Visit Resource]({resource.get('link', 'https://www.fema.gov')})
                """)
                st.markdown("---")
