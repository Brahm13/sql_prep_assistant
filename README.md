# SQL Interview Prep Assistant - RAG-based Search System

## Project Overview
This project implements an **SQL Interview Preparation Assistant** using **Retrieval-Augmented Generation (RAG)**. The assistant provides users with answers to SQL-related questions, leveraging three different search methods:

1. **Index-based Search** using MiniSearch.
2. **Hybrid Search** combining index-based and TF-IDF vector similarity.
3. **RAG Search** using OpenAI's GPT-based completion to generate answers.

The primary purpose of this assistant is to help users prepare for SQL technical interviews by answering SQL-related queries using a predefined dataset. The project uses a combination of keyword searches, hybrid methods, and generative AI for response generation.

## Features Completed
- **Search Methods**: Three methods (Index-based, Hybrid, RAG) implemented and working.
- **Data Ingestion Pipeline**: Set up with SQLAlchemy to store query and response data.
- **Chat Interface**: Allows users to ask questions, receive responses, and review previous sessions.
- **Conversation Management**: Allows a conversation history to be maintained across multiple queries until a new chat is started.
- **Query Monitoring**: Basic infrastructure in place for storing and displaying query history (yet to be fully tested).
- **Evaluation**: Model evaluation metrics for different search methods.

## To-Do
- **Testing**: Interface and query monitoring functionality need to be tested for full reproducibility.
- **Reproducibility & Containerization**: The project is not yet containerized (Docker) and reproducibility has been limited to running in Jupyter Notebooks.
- **Monitoring & Dashboard**: Complete and test query monitoring via the web interface.
- **Reproducibility**: Running the project through Jupyter Notebooks for now, though full environment reproducibility will be a future step.

---

## Project Setup & How to Run

### 1. Prerequisites
To run this project, ensure you have the following installed:
- **Python 3.8+**
- **Jupyter Notebook** (for running code in notebook format)
- **Streamlit** (`pip install streamlit`)
- **SQLAlchemy** (`pip install sqlalchemy`)
- **OpenAI** (`pip install openai`)

### 2. Running in Jupyter Notebooks (Reproducibility)
While full containerization is a to-do, you can run the project and inspect the results using Jupyter Notebook. Use the following steps:

1. Clone this repository.
2. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
3. Navigate to the Jupyter Notebook folder:
   ```bash
   cd notebooks
4.  Launch Jupyter Notebook:
    `jupyter notebook` 
5.  Open the relevant notebook (`data_ingestion_pipeline.ipynb`, `rag_search.ipynb`, etc.) and execute the cells sequentially to view the results.

### 3. Running with Streamlit (Chat Interface)

To run the chat interface locally using Streamlit, follow these steps:

1.  Ensure all dependencies are installed:
    `pip install -r requirements.txt` 
2.  Run the Streamlit app:
    `streamlit run assistant.py` 
3.  A local web server will start. Open the provided URL (usually http://localhost:8501/) in your web browser.

### 4. Database Setup
The project uses SQLite to store and track all user queries and responses. The database setup is simple and will be automatically created as `chat_sessions.db` in the project directory.

The key elements stored include:

-   **User ID**: A unique identifier for users (e.g., IP address or device ID).
-   **Question**: The SQL question posed by the user.
-   **Answer**: The assistant's response to the question.


## Search Method Evaluation

The performance of the search methods is evaluated using **Hit Rate** and **MRR (Mean Reciprocal Rank)**. Below are the evaluation results for different search strategies:

1.  **Index-Based Search (MiniSearch)**:
    
    -   Without Boost:
        -   Hit Rate: `0.1527`
        -   MRR: `0.0975`
    -   With Boost:
        -   Hit Rate: `0.0772`
        -   MRR: `0.0772`
2.  **Hybrid Search** (Combines Index-Based and TF-IDF Vector Similarity):
    
    -   Hit Rate: `0.1571`
    -   MRR: `0.0666`
3.  **RAG Evaluation**:
    
    -   Relevance:
        -   RELEVANT: `99%`
        -   PARTLY RELEVANT: `1%`

These evaluations indicate that the **Index-Based Search (without boost)** and **Hybrid Search** methods provide decent results, with **RAG** providing highly relevant answers for SQL questions.

----------

## Features Implemented

1.  **Search Methods**:
    
    -   **Index-Based Search**: Uses MiniSearch to retrieve relevant SQL questions and answers.
    -   **Hybrid Search**: Combines keyword matching with TF-IDF vector similarity for improved relevance.
    -   **RAG Search**: Utilizes OpenAI GPT models to generate SQL question answers by retrieving relevant context.
2.  **Query Ingestion and Monitoring**:
    
    -   Queries are stored in an SQLite database for tracking and analysis.
    -   Future plans include a monitoring dashboard to visualize query history.
3.  **Conversation Management**:
    
    -   Users can ask multiple questions in a session, and responses are stored as a single conversation.
    -   A "New Chat" button clears the conversation history for a new session.
4.  **Evaluation Metrics**:
    
    -   Hit Rate and MRR are used to evaluate search method effectiveness.
    -   Relevance scores demonstrate the high accuracy of the RAG-based assistant.

----------

## To-Do List

While the project is functional, some aspects remain incomplete and will be addressed in future iterations:

1.  **Interface & Monitoring**:
    
    -   The query monitoring dashboard is set up but needs thorough testing to ensure proper functionality.
2.  **Reproducibility**:
    
    -   Full reproducibility via Docker/containerization is yet to be implemented. Currently, the project can be reproduced using Jupyter Notebooks.
3.  **Deployment**:
    
    -   While the assistant works locally using Streamlit, deployment options (e.g., on the cloud) have not been implemented yet.

----------

## Conclusion

This project showcases a working **SQL Interview Preparation Assistant** using **RAG**, index-based search, and hybrid search techniques. It provides a foundation for future improvements such as enhanced monitoring, full containerization, and deployment. The current results demonstrate strong relevance in answering SQL-related questions, with scope for further optimization.