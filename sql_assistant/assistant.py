import os
import streamlit as st
from sqlalchemy.orm import sessionmaker
from ingestion_pipeline import engine, ChatSession, create_database
from rag import get_search_result
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

# SQLAlchemy session setup
Session = sessionmaker(bind=engine)
create_database()

# Streamlit UI
st.title("SQL Interview Prep Assistant")

# Sidebar for search method selection
search_method = st.sidebar.selectbox("Choose Search Method", ["RAG", "Index-Based", "Index-Based-Boosted", "Hybrid"])

# Global variables for conversation management
if 'conversation_id' not in st.session_state:
    st.session_state['conversation_id'] = 1  # Initialize conversation ID

if 'chat' not in st.session_state:
    st.session_state['chat'] = []  # Initialize chat session

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''  # Initialize input state

# Input area for questions
user_input = st.text_area("Ask your question here:", value=st.session_state['user_input'])

# Monitor dashboard link (will open a new window)
st.sidebar.markdown("[Monitor Dashboard](#)")

# Button to start a new chat
if st.sidebar.button("New Chat"):
    st.session_state['conversation_id'] += 1  # Increment conversation ID
    st.session_state['chat'] = []  # Clear current chat

# Handle response
if st.button("Get Response"):
    if user_input:
        with st.spinner("Fetching response..."):
            # Get response based on the selected method
            response = get_search_result(user_input, method=search_method.lower())
            
            # Display response
            st.markdown(f"**Assistant**: {response}")

            # Store the query and response in the session chat log
            st.session_state['chat'].append((user_input, response))

            # Store the query and response in the database
            user_id = os.getenv("USER_ID", "anonymous")  # Use environment variable or default value
            with Session() as session:
                new_session = ChatSession(
                    user_id=user_id, 
                    conversation_id=st.session_state['conversation_id'], 
                    question=user_input[:100],  # First 100 characters of the query
                    answer=response
                )
                session.add(new_session)
                session.commit()

            # Clear the user input field after the response is displayed
            st.session_state['user_input'] = ''  # Clear input for next query

# Display chat history as a single conversation until "New Chat" is clicked
st.sidebar.header("Conversation History")
for question, answer in st.session_state['chat']:
    st.sidebar.markdown(f"**You**: {question[:100]}")  # Only show first 100 chars in sidebar
    st.sidebar.markdown(f"**Assistant**: {answer}")

# Admin dashboard (removed password part for public monitoring)
if st.sidebar.button("View Admin Dashboard"):
    with Session() as session:
        chats = session.query(ChatSession).all()

    st.subheader("Admin Dashboard")
    for chat in chats:
        st.write(f"Conversation {chat.conversation_id}: {chat.question} -> {chat.answer}")
