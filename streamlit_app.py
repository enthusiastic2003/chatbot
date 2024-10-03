import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 Finance Chatbot")
st.write(
    "This is where the dashboard for our Finance Project will go.\n The Team Members are: \n 1. Sirjan \n 2. Rohit \n 3. Balaji \n 4. Suhas \n 5. Pratyush \n 6. Ishaq"
)
# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
