import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from transformers import pipeline
import openstreetmap

# Set up the AI models
st.set_page_config(page_title="AI Genesis: Disaster Response", layout="wide")

@st.cache_resource
def load_models():
    # Load your models here (DistilBERT, BERT-NER, Llama3-70B)
    # Example of loading a model with Hugging Face Transformers
    sentiment_analysis = pipeline("sentiment-analysis")
    ner_model = pipeline("ner")
    return sentiment_analysis, ner_model

sentiment_analysis, ner_model = load_models()

# Display title
st.title("AI Genesis: Disaster Response")

# Sidebar inputs for disaster location or other parameters
st.sidebar.header("Disaster Response Parameters")
location = st.sidebar.text_input("Enter disaster location:", "New York")
response_time = st.sidebar.slider("Response Time", min_value=1, max_value=12, value=6, step=1)

# Get disaster-related data from APIs (example: SerpAPI, OpenStreetMap)
def fetch_disaster_data(location):
    serpapi_key = "YOUR_SERPAPI_KEY"
    search_query = f"disaster relief efforts in {location}"
    response = requests.get(f"https://serpapi.com/search?q={search_query}&api_key={serpapi_key}")
    data = response.json()
    return data['organic_results']

# Display fetched data
disaster_data = fetch_disaster_data(location)
st.subheader("Disaster Relief Information")
st.write(disaster_data)

# Use the sentiment analysis model to analyze disaster-related news
def analyze_sentiment(text):
    return sentiment_analysis(text)

# Use Named Entity Recognition (NER) to identify locations and organizations in the disaster data
def analyze_entities(text):
    return ner_model(text)

# Visualizations using Plotly and Folium
# Example of creating a plotly chart
df = pd.DataFrame({
    'Time': ['0:00', '1:00', '2:00', '3:00'],
    'Response Level': [30, 40, 45, 50]
})
fig = px.line(df, x='Time', y='Response Level', title="Disaster Response Level Over Time")
st.plotly_chart(fig)

# Example of creating a Folium map
m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)  # New York coordinates
marker_cluster = MarkerCluster().add_to(m)

for i in range(5):  # Add dummy markers for demo
    folium.Marker([40.7128 + i*0.01, -74.0060 + i*0.01], popup=f"Location {i+1}").add_to(marker_cluster)

st.subheader("Map View")
st.write("Disaster locations are marked on the map:")
st.pydeck_chart(m)

# Show results from sentiment analysis and NER on fetched disaster data
for result in disaster_data:
    text = result['title']
    sentiment = analyze_sentiment(text)
    entities = analyze_entities(text)
    
    st.subheader(f"News: {result['title']}")
    st.write(f"Sentiment: {sentiment[0]['label']} with a confidence of {sentiment[0]['score']:.2f}")
    st.write("Named Entities: " + ", ".join([entity['word'] for entity in entities]))

# Footer
st.markdown("---")
st.write("AI Genesis: Disaster Response | Powered by AI & Real-time Data")
