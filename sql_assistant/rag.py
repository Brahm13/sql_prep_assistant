from openai import OpenAI
import ingestion_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

client = OpenAI()
index = ingestion_pipeline.index_search()
index_boosted = ingestion_pipeline.index_search_boosted()
data = ingestion_pipeline.load_data()
documents = data.to_dict(orient="records")


# RAG flow implementation using OPENAI API and index query using minsearch
def search(query):
    results = index_search(query)
    return results


prompt_template = """
You're a teaching assistant for SQL interview prep. Answer the QUESTION based on the CONTEXT from the database.
Use only the facts from the CONTEXT to answer the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

entry_template = """
question: {question}
answer: {answer}
category: {category}
difficulty_level: {difficulty_level}
tags: {tags}
example_query: {example_query}
explanation: {explanation}
common_mistakes: {common_mistakes}
related_questions: {related_questions}
""".strip()


def build_prompt(query, search_results):
    context = "\n\n".join([entry_template.format(**doc) for doc in search_results])
    prompt = prompt_template.format(question=query, context=context)
    return prompt


def llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


# query = "How do I create a CTE in SQL?"
# print(rag(query))


def rag_search(query):
    # return "RAG response to query: " + rag(query) + query
    return rag(query)


def index_search(query):
    results = index.search(query, num_results=10)
    # return results + query
    return results


def index_search_boosted(query):
    boost_dict = {"question": 3.0, "answer": 2.0, "tags": 1.8}
    results = index.search(
        query=query, filter_dict={}, boost_dict=boost_dict, num_results=10
    )
    # return results + query
    return results


tfidf_vectorizer = TfidfVectorizer()
data["combined_text"] = (
    data["question"]
    + " "
    + data["answer"]
    + " "
    + data["tags"]
    + " "
    + data["explanation"]
)
tfidf_matrix = tfidf_vectorizer.fit_transform(data["combined_text"])
# tfidf_matrix = tfidf_vectorizer.fit_transform(data['question'])


def vector_search(query, num_results=10):
    query_vec = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    doc_indices = cosine_similarities.argsort()[-num_results:][::-1]
    return [documents[idx] for idx in doc_indices]


def hybrid_search(query):
    keyword_results = index_search(query)  # Text-based search
    vector_results = vector_search(query, num_results=5)
    combined_results = keyword_results + vector_results
    # cosine similarity for each result
    query_vec = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # (result, similarity) for sorting
    result_with_similarity = [
        (result, sim) for result, sim in zip(combined_results, similarities)
    ]
    # similarity score
    ranked_results = sorted(result_with_similarity, key=lambda x: x[1], reverse=True)
    # results without similarity scores
    return [result for result, _ in ranked_results[:10]]


def rewrite_query(query):
    rewrite_prompt = f"Rewrite this query to be more effective for SQL interview questions retrieval: '{query}'"
    rewritten_query = llm(rewrite_prompt)
    return rewritten_query


def advanced_rag(query):
    rewritten_query = rewrite_query(query)
    search_results = hybrid_search(rewritten_query)
    # search_results = hybrid_search(rewritten_query)
    prompt = build_prompt(rewritten_query, search_results)
    answer = llm(prompt)
    return answer


# query = "How to join two tables in SQL?"
# print(advanced_rag(query))


def hybrid_search(query):
    # return "Hybrid search result for query: " + advanced_rag(query) + query
    return advanced_rag(query)


# Wrapper function to handle different search methods
def get_search_result(query, method="rag"):
    if method == "rag":
        return rag_search(query)
    elif method == "index-based":
        return index_search(query)
    elif method == "hybrid":
        return hybrid_search(query)
    elif method == "index-based-boosted":
        return index_search_boosted(query)
