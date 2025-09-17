import streamlit as st
from deep_translator import GoogleTranslator
from transformers import pipeline
import re
import plotly.graph_objects as go

# Initialize NLP classifier
nlp_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Sample groundwater data
gw_data = {
    "BlockA": {
        "2020": {"recharge": 130, "extraction": 110, "status": "Safe"},
        "2021": {"recharge": 140, "extraction": 115, "status": "Safe"},
        "2022": {"recharge": 135, "extraction": 120, "status": "Semi-Critical"},
        "2023": {"recharge": 145, "extraction": 125, "status": "Semi-Critical"},
    }
}

def translate_text(text, src_lang='auto', dest_lang='en'):
    return GoogleTranslator(source=src_lang, target=dest_lang).translate(text)

def parse_user_query(query):
    candidate_labels = ["recharge", "extraction", "status"]
    result = nlp_classifier(query, candidate_labels)
    parameter = result["labels"][0] if result["scores"][0] > 0.5 else None
    years = re.findall(r"\b(19|20)\d{2}\b", query)
    year = years[0] if years else None
    words = query.split()
    unit = None
    for word in reversed(words):
        if word.istitle() and word.lower() not in candidate_labels:
            unit = word
            break
    return {"parameter": parameter, "year": year, "unit": unit}

def get_groundwater_info(unit, year, parameter):
    if unit not in gw_data or year not in gw_data[unit]:
        return None
    return gw_data[unit][year].get(parameter)

def plot_recharge_trend(unit):
    data = gw_data.get(unit)
    if not data:
        return None
    years_sorted = sorted(data.keys())
    recharge_values = [data[y]["recharge"] for y in years_sorted]
    fig = go.Figure(data=go.Scatter(x=years_sorted, y=recharge_values, mode='lines+markers'))
    fig.update_layout(title=f"Recharge Trend in {unit}", xaxis_title="Year", yaxis_title="Recharge (mm)")
    return fig

st.title("INGRES Groundwater Data Chatbot with Multilingual Support")

input_lang = st.selectbox("Select your language:", ["auto", "hi", "en", "ta", "bn", "te"])

user_query = st.text_area("Enter your groundwater query:")

if st.button("Submit"):
    translated_query = translate_text(user_query, src_lang=input_lang, dest_lang='en')
    st.write(f"Translated query (to English): {translated_query}")

    parsed = parse_user_query(translated_query)
    st.write(f"Parsed query: {parsed}")

    info = get_groundwater_info(parsed["unit"], parsed["year"], parsed["parameter"])
    if info is None:
        response_en = "Requested data not found."
    else:
        response_en = f"{parsed['parameter'].capitalize()} in {parsed['unit']} for {parsed['year']}: {info}"
    st.write(f"Response (in English): {response_en}")

    if input_lang != 'en' and input_lang != 'auto':
        response_translated = translate_text(response_en, src_lang='en', dest_lang=input_lang)
        st.write(f"Response (in your language): {response_translated}")

    if parsed["parameter"] == "recharge" and info is not None:
        fig = plot_recharge_trend(parsed["unit"])
        if fig:
            st.plotly_chart(fig)

