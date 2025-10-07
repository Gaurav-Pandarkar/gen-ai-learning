from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

llm = ChatOllama(
    model = 'Llama3.2',
    max_tokens=500
)

while True:
    prompt = input("User:")
    if prompt.lower()=='exit':
        break
    response = llm.invoke(prompt)
    print("Bot Response:",response.content)
print("GoodBye!!!")
