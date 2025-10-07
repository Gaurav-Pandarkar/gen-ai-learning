import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                  model="gpt-4o", 
                max_tokens=300)

prompt = "What is Java ?"

response = llm.invoke(prompt)
print(response.content)