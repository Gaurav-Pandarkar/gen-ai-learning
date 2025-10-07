from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model = 'Llama3.2',
    max_tokens=100
)

prompt = "What is the capital of India ?"

response = llm.invoke(prompt)
print(response)