import streamlit as st
import requests
import sqlite3
import pandas as pd

import streamlit_authenticator as stauth

# Check if the user is logged out
if "page" not in st.session_state:
    st.session_state.page = "login"

# If the user is logged out, show a message and stop further execution
if st.session_state["page"] == "logged_out":
    st.title("👋 You have been logged out")
    st.success("Thanks for visiting the Retail Chatbot.")
    st.markdown("To start again, please reload or [click here to login again].")
    st.stop()

# Define the API endpoint
API_URL = "http://127.0.0.1:8000/chat"  # Your FastAPI endpoint

# Set up the Streamlit app
st.set_page_config(page_title="Retail Chatbot", page_icon="🛒")
st.title("Retail RAG Chatbot")


# Authenticate the user
passwords = ['123', 'abc']
hashed_passwords = stauth.Hasher(passwords).generate()
# st.warning(hashed_passwords)

# Define users with hashed passwords
# This is a stub for user authentication. In a real app, you would fetch this from a database or secure store.
# Here, we are using a simple dictionary for demonstration purposes.
users = {
    "usernames": {
        "john": {"name": "John", "password": hashed_passwords[0]},
        "sarah": {"name": "Sarah", "password": hashed_passwords[1]},
    }
}


# Initialize the authenticator
authenticator = stauth.Authenticate(
    users, "retail_chat", "abcdef", cookie_expiry_days=0
)


# Display the login form
name, auth_status, username = authenticator.login(form_name="Login", location="main")

# Handle incorrect credentials

if auth_status is False:
    st.error("Username/password is incorrect.")
    st.stop()

# Handle no credentials entered yet
elif auth_status is None:
    st.warning("Please enter your credentials.")
    st.stop()

# If authenticated, show chat UI and logout option
else:
    st.sidebar.success(f"Logged in as {name}")

    if st.sidebar.button("Logout", key="logout_button"):
        authenticator.logout("Logout", location="sidebar")
        # Force rerun and mark user as logged out
        st.session_state.page = "logged_out"
        st.rerun()

# Initialize session state for authenticated user
if "authenticated" not in st.session_state:
    st.session_state.authenticated = True  # Stub for now
    st.session_state.username = username

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

# Input for user query
query = st.chat_input("Ask about your order, inventory, or support...")

# If the user submits a query, send it to the API and display the response
if query:
    st.session_state.messages.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)
    
    
    headers = {"x-user": st.session_state.username}
    # Prepare payload with user and query
    payload = {"query": query}

    # Send the query to the API
    response = requests.post(API_URL, json=payload, headers=headers)
    # st.write("DEBUG:", response.status_code, response.text)


    if response.ok:
        answer = response.json().get("response", "No response.")
    else:
        answer = "API error. Check logs."

    st.session_state.messages.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)

def get_chat_history(user: str):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    
    # Ensure the table exists
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user TEXT,
                        message TEXT,
                        answer TEXT
                    )''')
    
    # Now it's safe to run the SELECT
    df = pd.read_sql_query(
        "SELECT message, answer FROM chat_history WHERE user = ? ORDER BY id DESC",
        conn,
        params=(user,)
    )
    
    conn.close()
    return df

with st.expander("📜 View Chat History"):
    if st.button("Load Chat History"):
        chat_df = get_chat_history(st.session_state.username)
        if chat_df.empty:
            st.info("No past chats found.")
        else:
            st.dataframe(chat_df, use_container_width=True)
