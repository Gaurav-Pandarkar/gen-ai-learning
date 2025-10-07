import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage


llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                  model="gpt-4o", 
                  max_tokens=300
                )

messages = []

while True:
    prompt = input("User :")
    if prompt.lower()=="exit":
        break
    messages.append(HumanMessage(prompt))
    response=llm.invoke(messages)
    messages.append(response.content)

    print("Bot Output:"+response.content)

print("GoodBye!!")
