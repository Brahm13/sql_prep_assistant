from openai import OpenAI
import csv
import os



client = OpenAI()



def generate_sql_questions(number_of_questions):
    generated_questions = []
    existing_questions = set()
    for i in range(number_of_questions):
        prompt = """
Generate a scenario-based SQL interview question suitable for a data-related role (like Data Analyst, Data Engineer, Business Analyst, or Business Intelligence). 
The question should require critical thinking and involve a complex SQL operation such as joins, subqueries, aggregations, recursive queries, or window functions.

For each question, please include:
1. The SQL answer that effectively solves the question.
2. The intended difficulty level (Easy, Intermediate, Advanced).
3. Relevant tags indicating the job role for which the question may be asked (e.g., Data Analyst, Data Engineer) and other appropriate SQL tag (e.g. Join, Recursive Join, Aggregate function, CTE etc.)
4. A brief explanation of the SQL code provided.
5. Common mistakes candidates might make while answering.
6. Related questions that could be asked in conjunction.

Make sure to format the output as a single string that includes all components separated by semicolons in the following format:
```
"id";"question";"answer";"category";"difficulty_level";"tags";"example_query";"explanation";"common_mistakes";"related_questions"
```
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use "gpt-4" if you have access
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant specialized in generating structured datasets.",
                },
                {"role": "user", "content": prompt},
            ],
            # temperature=1,
            # max_tokens=2048,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
        )
        question_text = response.choices[0].message.content #Access the content correctly
        if question_text not in existing_questions:  # Check for uniqueness
            existing_questions.add(question_text)  
            question_id = len(generated_questions) + 1                    
            parts = question_text.strip().split(";")
            if len(parts) >= 10:  
                generated_questions.append(
                    {
                        "id": question_id,
                        "question": parts[1],
                        "answer": parts[2],
                        "category": parts[3],
                        "difficulty_level": parts[4],
                        "tags": parts[5],
                        "example_query": parts[6],
                        "explanation": parts[7],
                        "common_mistakes": parts[8],
                        "related_questions": parts[9],
                    }
                )
    return generated_questions



def save_to_csv(questions, filename="sql_interview_questions.csv"):
    keys = questions[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys, delimiter=";")
        dict_writer.writeheader()
        dict_writer.writerows(questions)



if __name__ == "__main__":
    questions = generate_sql_questions(500)  # Adjust for desired number
    save_to_csv(questions)
