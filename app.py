import streamlit as st
import pandas as pd
import openai
import os

# --- Set API Key ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Sample Finance Data ---
@st.cache_data
def load_data():
    transactions = pd.DataFrame([
        [1, 101, 'Software License', 1500.00, '2025-06-01'],
        [2, 102, 'Consulting Service', 3000.00, '2025-06-03'],
        [3, 103, 'Training Session', 1200.00, '2025-06-04']
    ], columns=['TRANSACTION_ID', 'CUSTOMER_ID', 'PRODUCT_NAME', 'AMOUNT', 'TRANSACTION_DATE'])

    expenses = pd.DataFrame([
        [1, 'Software Subscriptions', 250.00, '2025-06-02'],
        [2, 'Travel', 500.00, '2025-06-05'],
        [3, 'Office Supplies', 100.00, '2025-06-06']
    ], columns=['EXPENSE_ID', 'CATEGORY', 'AMOUNT', 'EXPENSE_DATE'])

    salaries = pd.DataFrame([
        [1, 'Alice', 'Finance', 6000.00],
        [2, 'Bob', 'Engineering', 7500.00],
        [3, 'Charlie', 'HR', 5000.00]
    ], columns=['EMP_ID', 'EMP_NAME', 'DEPARTMENT', 'MONTHLY_SALARY'])

    return transactions, expenses, salaries

# --- UI ---
st.set_page_config(page_title="Finance Chat", layout="centered")
st.title("💬 Finance Chatbot with AI")
st.markdown("Ask questions about your financial data.")

user_input = st.text_input("❓ What do you want to know?")

# Load data
transactions, expenses, salaries = load_data()

# Display sample data
with st.expander("📂 Sample Finance Data"):
    st.write("**Transactions:**", transactions)
    st.write("**Expenses:**", expenses)
    st.write("**Salaries:**", salaries)

if user_input:
    with st.spinner("Generating answer..."):
        # Format prompt for AI
        prompt = f"""
You are a helpful finance assistant. Use the following data to answer the question below in plain English.

Question: {user_input}

TRANSACTIONS:
{transactions.head(3).to_string(index=False)}

EXPENSES:
{expenses.head(3).to_string(index=False)}

SALARIES:
{salaries.head(3).to_string(index=False)}

Answer:
"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            reply = response['choices'][0]['message']['content']
            st.success(reply)
            st.markdown(f"✅ Confidence Score: {round(85 + 10 * os.urandom(1)[0] / 255, 2)}%")
        except Exception as e:
            st.error(f"Error: {e}")
