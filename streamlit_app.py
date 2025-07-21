import streamlit as st
import requests
import sqlite3
import pandas as pd

import streamlit_authenticator as stauth

if "page" not in st.session_state:
    st.session_state.page = "login"

if st.session_state["page"] == "logged_out":
    st.title("👋 You have been logged out")
    st.success("Thanks for visiting the Retail Chatbot.")
    st.markdown("To start again, please reload or [click here to login again].")
    st.stop()

API_URL = "http://127.0.0.1:8000/chat"  # Your FastAPI endpoint

st.set_page_config(page_title="Retail Chatbot", page_icon="🛒")
st.title("Retail RAG Chatbot")



passwords = ['123', 'abc']
hashed_passwords = stauth.Hasher(passwords).generate()
# st.warning(hashed_passwords)
# Dummy credentials
users = {
    "usernames": {
        "john": {"name": "John", "password": hashed_passwords[0]},
        "sarah": {"name": "Sarah", "password": hashed_passwords[1]},
    }
}

authenticator = stauth.Authenticate(
    users, "retail_chat", "abcdef", cookie_expiry_days=0
)

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
        # Clear all session state variables
        # for key in list(st.session_state.keys()):
        #     del st.session_state[key]

        # Force rerun and mark user as logged out
        st.session_state.page = "logged_out"
        st.rerun()





if "authenticated" not in st.session_state:
    st.session_state.authenticated = True  # Stub for now
    st.session_state.username = username

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

query = st.chat_input("Ask about your order, inventory, or support...")

if query:
    st.session_state.messages.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)
    
    payload = {
    "user": st.session_state.username,
    "query": query
    }

    # st.write("Sending payload:", payload)
    response = requests.post(API_URL, json=payload)

    if response.ok:
        answer = response.json().get("response", "No response.")
    else:
        answer = "API error. Check logs."

    st.session_state.messages.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)

def get_chat_history(user: str):
    conn = sqlite3.connect("chat_history.db")
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
