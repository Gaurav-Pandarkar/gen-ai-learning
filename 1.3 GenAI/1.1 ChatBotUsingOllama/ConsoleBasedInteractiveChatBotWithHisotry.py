from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

llm=ChatOllama(
    model = 'Llama3.2',
    max_tokens = 500,
    temperture = 0.7,
    n=2
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
