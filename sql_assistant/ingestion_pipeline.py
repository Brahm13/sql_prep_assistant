import os
import pandas as pd
import minsearch
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os


Base = declarative_base()
engine = create_engine("sqlite:///chat_sessions.db")
Session = sessionmaker(bind=engine)
path = "../data/Cleaned_dataset.csv"


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    conversation_id = Column(Integer)
    question = Column(Text)
    answer = Column(Text)

def load_env():
    load_dotenv()  
    openai_api_key = os.getenv('OPENAI_API_KEY')
    print(openai_api_key)  


def create_database():
    Base.metadata.create_all(engine)


def load_data():
    return pd.read_csv(path)


def index_search():
    data = load_data()
    documents = data.to_dict(orient="records")
    index = minsearch.Index(
        [
            "question",
            "answer",
            "category",
            "difficulty_level",
            "tags",
            "example_query",
            "explanation",
            "common_mistakes",
            "related_questions",
        ],
        keyword_fields=["question", "answer"],
    )
    index.fit(documents)
    return index


# index search with boost
def index_search_boosted(
    # query, num_results=10, boost_dict={"question": 3.0, "answer": 2.0, "tags": 1.8}
):
    data = load_data()
    documents = data.to_dict(orient="records")
    index = minsearch.Index(
        [
            "question",
            "answer",
            "category",
            "difficulty_level",
            "tags",
            "example_query",
            "explanation",
            "common_mistakes",
            "related_questions",
        ],
        keyword_fields=["question", "answer"],
    )
    index.fit(documents)
    return index


if __name__ == "__main__":
    create_database()
    load_env()
