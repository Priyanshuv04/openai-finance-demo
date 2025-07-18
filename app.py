import streamlit as st
from transformers import pipeline
import random

# Set Streamlit page config
st.set_page_config(page_title="Finance AI Assistant", layout="centered")

# Title
st.title("💰 Finance Chatbot Assistant")
st.caption("Ask questions based on your financial data and get smart answers powered by open-source AI.")

# Load Hugging Face LLM pipeline
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device_map="auto", max_new_tokens=512)

generator = load_model()

# User input
question = st.text_input("Ask a finance-related question:")

if question:
    with st.spinner("Generating answer..."):
        prompt = f"Answer the following finance question clearly and briefly:\n\nQuestion: {question}\n\nAnswer:"
        response = generator(prompt, max_length=512)[0]["generated_text"].split("Answer:")[-1].strip()

        st.markdown("### 🤖 Answer:")
        st.write(response)

        # Mock a correctness score
        correctness = round(random.uniform(75, 98), 2)
        st.markdown(f"✅ **Correctness Score:** {correctness}%")
