# sql_prep_assistant
This is a RAG application project built as part of LLM Zoomcamp for using ai to prep for SQL interview questions.

### To be deleted:
```
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4o-mini", #"gpt-4o-mini-2024-07-18"
  messages=[],
  temperature=1,
  max_tokens=2048,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  response_format={
    "type": "text"
  }
)
```

