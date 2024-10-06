import os
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine('sqlite:///chat_sessions.db')  # Using SQLite for simplicity
Session = sessionmaker(bind=engine)

# Define the ChatSession model
class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    id = Column(Integer, primary_key=True)
    user_id = Column(String)  # User ID or browser ID
    conversation_id = Column(Integer)  # ID for chat sessions to group queries
    question = Column(Text)  # Store the query
    answer = Column(Text)  # Store the response

# Create the database tables if they don't exist
def create_database():
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    create_database()
