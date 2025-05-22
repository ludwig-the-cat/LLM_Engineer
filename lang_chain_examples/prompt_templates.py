# Импорт шаблонов сообщений (PromptTemplate) и промтов для системного и пользовательского сообщений
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate
)

# Импорт Ollama LLM-клиента
from langchain_ollama import ChatOllama

# Импорт параметров подключения: адрес сервера Ollama и название модели
from tools.options import base_url, model

# Инициализация LLM через Ollama (локальный или удалённый сервер, и выбранная модель, например "mistral")
llm = ChatOllama(base_url=base_url, model=model)

# Создание шаблона пользовательского сообщения с параметрами topics и points
# При вызове шаблона эти параметры будут заменены на конкретные значения
question = HumanMessagePromptTemplate.from_template(
    'tell about the {topics} in {points} points'
)

# Создание шаблона системного сообщения, где задаётся "роль" модели — учитель из определённой школы
system = SystemMessagePromptTemplate.from_template(
    'You are {school} teacher. You answer in short sentences'
)

# Объединение системного и пользовательского сообщений в один шаблон диалога
messages = [system, question]

# Обёртка всех сообщений в единый `ChatPromptTemplate` — шаблон, который можно заполнить словарём переменных
template = ChatPromptTemplate(messages)

# Подстановка конкретных значений в шаблон:
# 'grim' — это стиль учителя, 'night' — тема, '5' — количество пунктов
full_question = template.invoke({
    'school': 'grim',
    'topics': 'night',
    'points': 5
})

# Отправка сформированного промта в LLM
response = llm.invoke(full_question)

# Вывод только текстовой части ответа
print(response.content)
