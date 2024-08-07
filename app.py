# Install necessary libraries
# pip install openai pinecone-client
import streamlit as st
import openai
import pinecone
import fitz  # PyMuPDF

# Initialize OpenAI
openai.api_key = 'sk-proj-ci0nnLajRdB2nRR2grvnT3BlbkFJCfcE3rbrm5TY7itajPYS'

# Initialize Pinecone
pinecone.init(api_key='1e1a7725-b030-4b30-aa95-dadd43cf2548')

# Create a Pinecone index
#pinecone.create_index('cv-analysis', dimension=128)

# Connect to the index
index = pinecone.Index('cv-analysis')

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_info_from_cv(cv_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that extracts key information from CVs."},
            {"role": "user", "content": f"Extract key information from the following CV:\n\n{cv_text}"}
        ],
        max_tokens=500
    )
    return response.choices[0].message['content'].strip()

def predict_personality(extracted_info):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that predicts personality traits based on CV information."},
            {"role": "user", "content": f"Based on the following information, predict the personality traits:\n\n{extracted_info}"}
        ],
        max_tokens=500
    )
    return response.choices[0].message['content'].strip()

def store_data(id, data):
    index.upsert([(id, data)])

def retrieve_data(id):
    result = index.fetch([id])
    return result

def process_cv(pdf_file):
    cv_text = extract_text_from_pdf(pdf_file)
    extracted_info = extract_info_from_cv(cv_text)
    personality_traits = predict_personality(extracted_info)
    return extracted_info, personality_traits

# Streamlit UI
st.title("Personality prediction system via CV")

uploaded_file = st.file_uploader("Upload your CV (PDF)", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing..."):
        extracted_info, personality_traits = process_cv(uploaded_file)
    
    st.subheader("Extracted Information")
    st.text(extracted_info)
    
    st.subheader("Predicted Personality Traits")
    st.text(personality_traits)