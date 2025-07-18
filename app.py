import streamlit as st
import pandas as pd
from openai import OpenAI

# Load OpenAI API key securely
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

st.set_page_config(page_title="Finance AI Chatbot", page_icon="💸")
st.title("💬 Finance AI Chatbot")
st.caption("Ask financial questions about your data (simulated).")

# Sample data for demo
transactions = pd.DataFrame({
    "TRANSACTION_ID": [1, 2, 3],
    "CUSTOMER_ID": [101, 102, 103],
    "PRODUCT_NAME": ["Software License", "Consulting Service", "Training Session"],
    "AMOUNT": [1500.00, 3000.00, 1200.00],
    "TRANSACTION_DATE": ["2025-06-01", "2025-06-03", "2025-06-04"]
})

question = st.text_input("Ask a financial question:")

if st.button("Submit") and question:
    with st.spinner("Generating answer..."):
        csv_context = transactions.to_csv(index=False)

        system_prompt = "You are a financial analyst. Based only on the data provided, answer the user's question clearly and accurately."

        user_prompt = f"""
Here is the transaction data:
{csv_context}

User question: {question}
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # changed here
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content
        st.markdown("### 📊 Answer")
        st.write(answer)
        st.markdown("---")
        st.caption("✅ Powered by GPT | Based on simulated data")
