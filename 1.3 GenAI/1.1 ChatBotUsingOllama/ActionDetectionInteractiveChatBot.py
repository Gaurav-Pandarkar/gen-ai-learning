from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="Llama3.2")

def detect_action(prompt: str):
    prompt_lower = prompt.lower()

    if any(word in prompt_lower for word in ["image", "picture", "draw", "generate image", "create image"]):
        return "IMAGE"
    elif any(word in prompt_lower for word in ["audio", "sound", "speech", "convert to audio"]):
        return "AUDIO"
    elif any(word in prompt_lower for word in ["translate", "translation", "from english", "to marathi"]):
        return "TRANSLATION"
    else:
        return "TEXT"

def main():
    user_input = input("Enter your prompt here: ")

    action_type = detect_action(user_input)

    detailed_prompt = f"The user wants to perform a {action_type.lower()} task. Original input: {user_input}"

    response = llm.invoke([HumanMessage(content=detailed_prompt)])

    print("Detected Action:", action_type)
    print("Response as per detected action", response.content)

if __name__ == "__main__":
    main()
