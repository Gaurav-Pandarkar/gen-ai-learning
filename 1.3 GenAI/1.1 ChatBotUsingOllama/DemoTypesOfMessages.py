from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

llm = ChatOllama(
    model='Llama3.2',
    max_tokens=1000
)

context = SystemMessage(
    content="You are c++ developer"
)

human_message = HumanMessage("Tell me joke")
response = llm.invoke([context,human_message])
print(response.content)

ai_message = AIMessage(response.content)

print('**********************************************')


context = SystemMessage(
    content = "You are java developer"
)

human_message = HumanMessage('Tell me joke')
response = llm.invoke([context,human_message,ai_message])
print(response.content)

