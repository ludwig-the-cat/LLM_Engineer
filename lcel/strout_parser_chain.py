# Импорт шаблонов сообщений и общего шаблона диалога
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

# Импорт парсера, который извлекает текстовую строку из ответа модели
from langchain_core.output_parsers import StrOutputParser

# Импорт клиента Ollama
from langchain_ollama import ChatOllama

# Импорт настроек: адрес Ollama-сервера и модель
from tools.options import base_url, model

# Создание объекта модели
llm = ChatOllama(base_url=base_url, model=model)

# Шаблон пользовательского сообщения (с параметрами для подстановки)
question = HumanMessagePromptTemplate.from_template(
    'tell about the {topics} in {points} points'
)

# Шаблон системного сообщения (роль модели — учитель определённого типа)
system = SystemMessagePromptTemplate.from_template(
    'You are {school} teacher. You answer in short sentences'
)

# Объединение system + user сообщений в общий шаблон диалога
messages = [system, question]
template = ChatPromptTemplate(messages)

# Создание цепочки: шаблон → модель → парсер
chain = template.pipe(llm).pipe(StrOutputParser())
# Альтернатива: chain = template | llm | StrOutputParser()

# Подставляем значения в шаблон и вызываем цепочку
response = chain.invoke({
    'topics': 'solar system',
    'points': 7,
    'school': 'conservative'
})

# Проверка типа результата — теперь это строка, а не AIMessage
print(response)        # Сам ответ от модели
