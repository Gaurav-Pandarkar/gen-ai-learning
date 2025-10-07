import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                  model="gpt-4o", 
                  max_tokens=300
                )
while True:
    prompt = input("User:")
    if prompt.lower()=='exit':
        break
    response = llm.invoke(prompt)
    print("Bot Response:",response.content)
print("GoodBye!!!")
