#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd


# In[6]:


data = pd.read_csv('../data/Cleaned_dataset.csv')


# ## Ingestion 
# ##### Text based or index [minsearch]

# In[31]:


# text based or index based search

import minsearch

documents = data.to_dict(orient='records')

def index_search(query, num_results=10,  boost_dict={'question': 3.0, 'answer': 2.0, 'tags':1.8}):
    index = minsearch.Index(
        ['question', 'answer', 'category', 'difficulty_level', 'tags', 
         'example_query', 'explanation', 'common_mistakes', 'related_questions'],
        keyword_fields=['question', 'answer']
        )
    index.fit(documents)
    return index.search(query, num_results=1,  boost_dict=boost_dict)


query = "Give me intermediate SQL interview questions"
search_results = index_search(query)
print(search_results)


# ##### Configure OPENAI API keys 

# In[19]:


import os
os.environ['OPENAI_API_KEY'] = 'xxxxxxx'


# In[20]:


# Configure OpenAI API for RAG
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{"role": "user", "content": query}]
)

response.choices[0].message.content


# ## RAG flow implementation using OPENAI API and index query using minsearch

# In[21]:


# RAG flow implementation using OPENAI API and index query using minsearch
def search(query):
   results = index.search(query, num_results=10)
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
       model='gpt-4o-mini',  
       messages=[{"role": "user", "content": prompt}]
   )
   return response.choices[0].message.content

def rag(query):
   search_results = search(query)
   prompt = build_prompt(query, search_results)
   answer = llm(prompt)
   return answer

query = "How do I create a CTE in SQL?"
print(rag(query))


# ## Hybrid Search

# In[32]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


tfidf_vectorizer = TfidfVectorizer()
data['combined_text'] = data['question'] + ' ' + data['answer'] + ' ' + data['tags'] + ' ' + data['explanation']
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_text'])
# tfidf_matrix = tfidf_vectorizer.fit_transform(data['question'])

def vector_search(query, num_results=10):
    query_vec = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    doc_indices = cosine_similarities.argsort()[-num_results:][::-1]
    return [documents[idx] for idx in doc_indices]


def hybrid_search(query):
    keyword_results = index.search(query, num_results=5)  # Text-based search
    vector_results = vector_search(query, num_results=5)  
    
    combined_results = keyword_results + vector_results
    
    # cosine similarity for each result
    query_vec = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # (result, similarity) for sorting
    result_with_similarity = [(result, sim) for result, sim in zip(combined_results, similarities)]
    
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
    search_results = hybrid_search(rewritten_query)
    prompt = build_prompt(rewritten_query, search_results)  
    answer = llm(prompt)  
    return answer

query = "How to join two tables in SQL?"
print(advanced_rag(query))


# ## Retrieval Evaluation

# In[23]:


from tqdm.auto import tqdm

df_question = pd.read_parquet('../data/retrieval_evaluate_dataset.parquet')

ground_truth = df_question.to_dict(orient='records')

# df_question.head()


# In[33]:


def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt += 1
    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)

def retrieval_search(query):
    # boost = {}
    results = index.search(
        query=query,
        # filter_dict={},
        # boost_dict=boost,
        num_results=10
    )
    return results


def evaluate_retrieval(ground_truth, search_function):
    relevance_total = []
    for q in tqdm(ground_truth):
        doc_id = q['id']
        results = search_function(q['question'])
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)
    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }

evaluation_result = evaluate_retrieval(ground_truth, lambda q: retrieval_search(q))
print(evaluation_result)


# In[35]:


evaluation_result = evaluate_retrieval(ground_truth, lambda q: index_search(q))
print(evaluation_result)


# In[34]:


evaluation_hybrid_search = evaluate_retrieval(ground_truth, lambda q: hybrid_search(q))
print(evaluation_hybrid_search)


# ## RAG Evaluation

# In[28]:


import json


# In[ ]:


df_sample = df_question.sample(n=20, random_state=1)
sample = df_sample.to_dict(orient='records')


prompt2_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()

# Function to evaluate RAG outputs using an LLM
def evaluate_rag(sample, rag_function, llm_evaluator):
    evaluations = []
    for record in tqdm(sample):
        question = record['question']
        answer_llm = rag_function(question)  # Generate answer from RAG
        
        # Prepare the evaluation prompt
        prompt = prompt2_template.format(
            question=question,
            answer_llm=answer_llm
        )
        
        # Evaluate using LLM 
        evaluation = llm_evaluator(prompt)
        evaluation = json.loads(evaluation)
        
        # Store evaluation results
        evaluations.append((record, answer_llm, evaluation))
    
    return evaluations


def rag_function(question):    
    return rag(question) + question

def llm_evaluator(prompt):
    evaluation = llm(prompt)
    return json.dumps(evaluation)

# Run the RAG evaluation
evaluations = evaluate_rag(sample, rag_function, llm_evaluator)



# In[55]:


# evaluations_ = evaluations
relevance_values = []
for eval_item in evaluations:
    json_str = eval_item[2]  # Get the JSON-like string
    relevance_data = json.loads(json_str)  # Parse it into a dictionary
    relevance_values.append(relevance_data['Relevance'])



# In[56]:


df_eval = pd.DataFrame(evaluations, columns=['record', 'answer', 'evaluation'])


df_eval['id'] = df_eval['record'].apply(lambda d: d['id'])
df_eval['question'] = df_eval['record'].apply(lambda d: d['question'])


df_eval['relevance'] = df_eval['evaluation'].apply(lambda d: json.loads(d)['Relevance'])
df_eval['explanation'] = df_eval['evaluation'].apply(lambda d: json.loads(d)['Explanation'])


df_eval.drop(columns=['record', 'evaluation'], inplace=True)

df_eval.to_csv('../data/rag-evaluation.csv', index=False)


print(df_eval['relevance'].value_counts(normalize=True))

