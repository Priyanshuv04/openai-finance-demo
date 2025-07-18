import streamlit as st
import pandas as pd
from openai import OpenAI

# Load OpenAI API key securely from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

st.set_page_config(page_title="Finance AI Chatbot", page_icon="💸")
st.title("💬 Finance AI Chatbot")
st.caption("Ask financial questions about your data (simulated).")

# Sample finance data (replace with Snowflake queries later)
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

        # Prepare CSV-style context for GPT
        csv_context = transactions.to_csv(index=False)

        system_prompt = "You are a financial analyst. Based only on the data provided, answer the user's question clearly and accurately."

        user_prompt = f"""
Here is the transaction data:
{csv_context}

User question: {question}
"""

        # Use GPT-4 model via OpenAI v1.0+ SDK
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        # Extract the answer
        answer = response.choices[0].message.content

        # Display result
        st.markdown("### 📊 Answer")
        st.write(answer)
        st.markdown("---")
        st.caption("✅ Powered by GPT-4 | Based on provided sample data")
