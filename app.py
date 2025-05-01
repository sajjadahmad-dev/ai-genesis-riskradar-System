import requests
import os
from transformers import pipeline
from groq import Groq
from serpapi import GoogleSearch

NER = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def fetch_news_data(query):
    search = GoogleSearch({
        "q": query,
        "num": 10,
        "api_key": os.getenv("SERPAPI_KEY")
    })
    results = search.get_dict().get("organic_results", [])
    return [{"title": r["title"], "link": r["link"], "source": r.get("source")} for r in results]

def classify_risks(articles):
    tags = []
    for article in articles:
        entities = NER(article["title"])
        extracted = [e['word'] for e in entities if e['entity_group'] in ["ORG", "MISC"]]
        tags.extend(extracted)
    return pd.DataFrame(articles), tags

def generate_response(tags, topic):
    tag_text = ", ".join(tags)
    prompt = f"""
    Analyze these tags: {tag_text} in the context of {topic}.
    Identify any business risks or opportunities and give a short strategic summary.
    """
    chat = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}]
    )
    return chat.choices[0].message.content.strip()
