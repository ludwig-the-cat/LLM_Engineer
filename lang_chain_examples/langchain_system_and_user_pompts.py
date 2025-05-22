# Импорт класса ChatOllama — обёртки вокруг LLM-интерфейса, работающего с Ollama (например, с моделями вроде LLaMA, Mistral и т.п.)
from langchain_ollama import ChatOllama

# Импорт типов сообщений: HumanMessage — от пользователя, SystemMessage — для задания "роли" модели
from langchain_core.messages import SystemMessage, HumanMessage

# Импорт настроек: адрес Ollama-сервера и название модели (например, 'mistral', 'llama2', 'phi', и т.п.)
from tools.options import base_url, model

# Инициализация объекта LLM с заданными параметрами
llm = ChatOllama(base_url=base_url, model=model)

# Создаём сообщение от пользователя (инструкция для модели)
question = HumanMessage('tell about the earth in 5 points')

# Задаём системное сообщение — это как "установка роли" для модели (влияет на стиль и тон ответа)
system = SystemMessage('You are teacher. You answer in short sentences')

# Объединяем оба сообщения в диалог: сначала системное, потом пользовательское
messages = [system, question]

# Отправляем цепочку сообщений в модель и получаем ответ
response = llm.invoke(messages)

# Выводим только текстовую часть ответа (объект `AIMessage` содержит поле `.content`)
print(response.content)
