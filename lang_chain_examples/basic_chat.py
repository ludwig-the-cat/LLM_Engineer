# Импорт основного класса LLM из LangChain, использующего Ollama в качестве backend'а
from langchain_ollama import ChatOllama

# Импорт параметров подключения к серверу Ollama: адрес и модель
from tools.options import base_url, model

# Создаём объект языковой модели (LLM), указывая адрес Ollama-сервера и используемую модель
llm = ChatOllama(base_url=base_url, model=model)

# Отправляем простой текстовый запрос в модель
response = llm.invoke('tell about the earth in 5 points')

# Печатаем только текстовый контент из ответа (response — это объект AIMessage)
print(response.content)
