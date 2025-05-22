from langchain_ollama import ChatOllama
from tools.options import BaseOptions

# Базовый вызов
llm = ChatOllama(base_url=BaseOptions.base_url, model=BaseOptions.model)
response = llm.invoke('tell about the earth in 5 points')
print(response.content)
