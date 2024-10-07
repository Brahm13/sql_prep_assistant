# SQL Interview Prep Assistant

This project is an end-to-end implementation of a SQL Interview Preparation Assistant, utilizing Retrieval-Augmented Generation (RAG) for answering SQL-related queries. It also integrates additional search methods (index-based and hybrid search) to support diverse query processing needs. This document outlines the project details, how to run it, and areas still under development.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [How to Run the Project](#how-to-run-the-project)
- [Functionality](#functionality)
- [Evaluation Results](#evaluation-results)
- [To-Do](#to-do)
- [Dependencies](#dependencies)

## Overview

The SQL Interview Prep Assistant is a tool designed to provide SQL interview questions and answers by searching a dataset and offering the most relevant response using three main search methods:
- **RAG (Retrieval-Augmented Generation)**
- **Index-based Search** (using `minisearch`)
- **Hybrid Search** (combining RAG and traditional methods)

The project is built using:
- **Python (v3.10)**
- **Streamlit** for the user interface
- **SQLAlchemy** for database management
- **Pipenv** for dependency management and reproducibility
- **OpenAI's GPT API** for generating RAG-based responses

## Project Structure

```bash
SQL_Interview_Prep_Assistant/
├── Pipfile
├── Pipfile.lock
├── README.md
├── data/
│   ├── Cleaned_dataset.csv
│   ├── ground-truth-retrieval.csv
│   ├── retrieval_evaluate_dataset.csv
│   ├── retrieval_evaluate_dataset.xlsx
│   ├── retrieval_evaluate_dataset.parquet
│   ├── Streamlit_UI.png  # Screenshot of the Streamlit interface
│   ├── rag-evaluation.csv
│   └── sql_question_answer_dataset.csv
├── notebooks/
│   ├── generate_question.ipynb
│   └── v1_rag_flow_and_evalution.ipynb
├── python_file/
│   └── dataset_generation.py
└── sql_assistant/
    ├── __pycache__/
    ├── assistant.py
    ├── chat_sessions.db
    ├── ingestion_pipeline.py
    ├── minsearch.py
    ├── rag.py
    └── v1_rag_flow_and_evalution.py


## How to Run the Project

1.  **Clone the repository:**
      
    `git clone <your-repo-url>
    cd SQL_Interview_Prep_Assistant` 
    
2.  **Set up the virtual environment and install dependencies:** Ensure you have `pipenv` installed. If not, you can install it via pip:
    
    `pip install pipenv` 
    
    Then, install the project dependencies:    
    `pipenv install` 
    
3.  **Activate the virtual environment:**    
    `pipenv shell` 
    
4.  **Run the Streamlit Interface:** To start the Streamlit app for interacting with the SQL Interview Prep Assistant, run the following command:
    `streamlit run sql_assistant/assistant.py` 
    
5.  **Jupyter Notebook for Reproducibility:** You can run the core logic and evaluations in the provided Jupyter notebooks inside the `notebooks/` directory:
    `jupyter notebook notebooks/v1_rag_flow_and_evalution.ipynb` 
    
    The evaluation results can also be viewed in the notebook.
    

## Functionality

-   **RAG-based Answering:** The assistant uses RAG to fetch SQL interview questions and their answers based on relevance.
-   **Search Methods:** Users can switch between three search methods via the UI:
    -   RAG-based
    -   Index-based (minisearch)
    -   Hybrid Search
-   **Conversation History:** Keeps track of questions and answers in a chat format.
-   **New Chat Feature:** Allows users to start fresh chats and stores previous conversations in a database.
-   **Monitoring Link:** A monitoring link is intended to display a dashboard of chat history, but this feature is currently not functional (see [To-Do](#to-do)).

## Evaluation Results

-   **Index-based Search (minisearch) without Boost:**
    
    -   Hit Rate: `0.1527`
    -   MRR (Mean Reciprocal Rank): `0.0975`
-   **Index-based Search (minisearch) with Boost:**
    
    -   Hit Rate: `0.0772`
    -   MRR: `0.0772`
-   **Hybrid Search:**
    
    -   Hit Rate: `0.1571`
    -   MRR: `0.0666`
-   **RAG Evaluation:**
    
    -   Relevance Scores:
        -   RELEVANT: `99%`
        -   PARTLY_RELEVANT: `1%`

## To-Do

The following areas are still under development or require improvements:

1.  **Monitoring Dashboard:** The monitor dashboard link currently throws an error (`OperationalError: no such table: chat_sessions`). This needs to be fixed.
2.  **Reproducibility & Containerization:** While Pipenv is used for dependency management, the project still lacks containerization (e.g., Docker).
3.  **Search Method Bug Fixes:** Only the RAG method is fully functional. Index-based and hybrid search methods require further debugging to work correctly.


## Dependencies

The project uses the following dependencies, managed via `pipenv`:
```toml
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
openai = "*"
scikit-learn = "*"
pandas = "*"
flask = "*"
jupyter = "*"
streamlit = "*"
sqlalchemy = "*"
python-dotenv = "*"

[dev-packages]
tqdm = "*"
ipywidgets = "*"

[requires]
python_version = "3.10"
```

## Running Tests and Evaluations

You can reproduce the search evaluations using the provided Jupyter notebooks. Simply open the notebook `v1_rag_flow_and_evalution.ipynb` and run the cells to evaluate search methods and RAG flow.

To visualize the results or modify the search algorithm, edit and run the notebooks in the `notebooks/` directory.

## Conclusion

This project implements an SQL Interview Prep Assistant using advanced search techniques, with RAG as the primary method. It also supports historical chat management and offers a Streamlit-based user interface. Some areas, like monitoring and containerization, are on to do list.
