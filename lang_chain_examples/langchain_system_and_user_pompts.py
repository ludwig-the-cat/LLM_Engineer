from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from tools.options import BaseOptions

# Применение langchain системный и пользовательский промт
llm = ChatOllama(base_url=BaseOptions.base_url, model=BaseOptions.model)
question = HumanMessage('tell about the earth in 5 points')
system = SystemMessage('You are teacher. You answer in short sentences')
messages = [system, question]
response = llm.invoke(messages)
print(response.content)