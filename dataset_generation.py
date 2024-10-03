from openai import OpenAI
import csv
import os
from collections import Counter
from tqdm import tqdm

client = OpenAI()


def generate_sql_questions(number_of_questions):
    generated_questions = []
    existing_questions = set()
    question_count = Counter()  # To keep track of occurrences of each question
    with tqdm(total=number_of_questions) as pbar:  # Initialize progress bar
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

Make sure to format the output as a single string that includes all components separated by semicolons in the following format::

"id";"question";"answer";"category";"difficulty_level";"tags";"example_query";"explanation";"common_mistakes";"related_questions"

The columns should have the following data types:
- **id**: Integer
- **question**: Text/String
- **answer**: Text/String
- **category**: Text/String (e.g., SQL)
- **difficulty_level**: Text/String (e.g., Intermediate, Advanced)
- **tags**: Array/List of Strings (e.g., CTE, SQL, Data Analysis)
- **example_query**: Text/String (an example SQL query)
- **explanation**: Text/String (a brief explanation of the answer)
- **common_mistakes**: Text/String (common errors made by candidates)
- **related_questions**: Array/List of Strings (questions related to the main question)
"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use the correct model version
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant specialized in generating structured datasets.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            question_text = response.choices[0].message.content  # Access correctly
            # Incrementing duplicate count
            question_count[question_text] += 1
            if (
                question_count[question_text] > 5
            ):  # Stop if duplicate exceeds 5 occurrences
                print(f"Duplicate question occurs more than 5 times: {question_text}")
                break
            if question_text not in existing_questions:  # Check for uniqueness
                existing_questions.add(question_text)
                question_id = len(generated_questions) + 1  # Counter for ID
                parts = question_text.strip().split(";")
                if len(parts) >= 10:  # Ensure the correct number of parts
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
                    # Save to CSV incrementally
                    save_to_csv(
                        [generated_questions[-1]],
                        filename="sql_interview_questions.csv",
                    )
            pbar.update(1)  # Update progress bar
    # After generation, print the highest to lowest duplicate questions
    print("\nDuplicate Question Counts:")
    for question, count in question_count.most_common():
        print(f"{count}: {question}")


def save_to_csv(questions, filename="sql_interview_questions.csv"):
    file_exists = os.path.isfile(filename)
    keys = questions[0].keys()
    with open(
        filename, "a", newline="", encoding="utf-8"
    ) as output_file:  # Append mode
        dict_writer = csv.DictWriter(output_file, fieldnames=keys, delimiter=";")
        if not file_exists:  # Write header only if file doesn't exist
            dict_writer.writeheader()
        dict_writer.writerows(questions)


# Main execution
if __name__ == "__main__":
    questions = generate_sql_questions(500)  # Adjust for desired number
