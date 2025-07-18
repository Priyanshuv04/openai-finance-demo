import streamlit as st
import pandas as pd
from transformers import pipeline

# Load the free Hugging Face model (no auth required)
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=256)

llm = load_model()

# Sample finance data — in a real app, fetch from Snowflake tables
@st.cache_data
def load_data():
    data = {
        "TRANSACTIONS": pd.DataFrame({
            "TRANSACTION_ID": [1, 2, 3],
            "CUSTOMER_ID": [101, 102, 103],
            "PRODUCT_NAME": ["Software License", "Consulting Service", "Training Session"],
            "AMOUNT": [1500.00, 3000.00, 1200.00],
            "TRANSACTION_DATE": ["2025-06-01", "2025-06-03", "2025-06-04"]
        }),
        "EXPENSES_LOG": pd.DataFrame({
            "EXPENSE_ID": [1, 2, 3],
            "CATEGORY": ["Software Subscriptions", "Travel", "Office Supplies"],
            "AMOUNT": [250.00, 500.00, 100.00],
            "EXPENSE_DATE": ["2025-06-02", "2025-06-05", "2025-06-06"]
        }),
        "EMPLOYEE_SALARIES": pd.DataFrame({
            "EMP_ID": [1, 2, 3],
            "EMP_NAME": ["Alice", "Bob", "Charlie"],
            "DEPARTMENT": ["Finance", "Engineering", "HR"],
            "MONTHLY_SALARY": [6000.00, 7500.00, 5000.00]
        })
    }
    return data

# Load Data
finance_data = load_data()

# Streamlit UI
st.title("💬 Finance Insights Chatbot (AI-Powered)")
st.write("Ask a question about your finance data:")

question = st.text_input("Type your question here...")

if question:
    # Convert the data to a string summary (simulate AI access to DB)
    combined_data_text = ""
    for name, df in finance_data.items():
        combined_data_text += f"\n\n{name}:\n{df.to_string(index=False)}"

    # Construct input prompt
    prompt = f"""
You are a financial analyst AI. Analyze the data below and answer the following question.

Data:{combined_data_text}

Question: {question}
Answer:
"""

    with st.spinner("Thinking..."):
        output = llm(prompt)[0]["generated_text"]
        # Extract only the part after "Answer:" to avoid repetition
        if "Answer:" in output:
            final_answer = output.split("Answer:")[-1].strip()
        else:
            final_answer = output.strip()

    # Display results
    st.markdown("### 💡 AI Answer")
    st.write(final_answer)

    st.markdown("#### ✅ Correctness Score")
    st.progress(0.8, text="Estimated 80% accurate (mock score)")
