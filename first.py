# Import necessary libraries
import fitz  # PyMuPDF for PDF text extraction
import openai  # OpenAI API
import pinecone  # Pinecone for vector database
from transformers import pipeline  # Hugging Face for personality prediction

# Initialize OpenAI API
openai.api_key = 'sk-proj-ci0nnLajRdB2nRR2grvnT3BlbkFJCfcE3rbrm5TY7itajPYS'

# Initialize Pinecone
pinecone.init(api_key='1e1a7725-b030-4b30-aa95-dadd43cf2548', environment='us-east-1')
index_name = 'cv-embeddings'
#pinecone.create_index(index_name, dimension=768)  # Dimension should match the embedding size
index = pinecone.Index(index_name)

print("hello")
print(index)
# Initialize Hugging Face pipeline for personality prediction
classifier = pipeline('fill-mask', model='bert-base-uncased')
print("hellohellohellohellohellohellohellohellohellohellohellohellohellohello")
# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to get embeddings from OpenAI
def get_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Function to store embeddings in Pinecone
def store_embedding(id, embedding):
    index.upsert([(id, embedding)])

# Function to predict personality using Hugging Face
def predict_personality(text):
    return classifier(text)

# Function to integrate all steps
def analyze_cv(file_path):
    # Step 1: Extract text
    text = extract_text_from_pdf(file_path)
    print("text:"+text)
    # Step 2: Get embeddings
    embeddings = get_embeddings(text)
    print("embeddings:")
    # Step 3: Store embeddings
    store_embedding(file_path, embeddings)
    print("store_embedding:")
    # Step 4: Predict personality
    personality_prediction = predict_personality(text)
    print("personality_prediction:"+personality_prediction)
    return personality_prediction

# Example usage
file_path = 'Docs/embedded-software-engineer-resume-example.pdf'
print("file_path")
print(file_path)
personality_traits = analyze_cv(file_path)
print("personality_traits")
print(personality_traits)
print(personality_traits)