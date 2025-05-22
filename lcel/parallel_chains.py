from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_ollama import ChatOllama
from tools.options import base_url, model

# Инициализируем модель Ollama с базовым URL и моделью из настроек
llm = ChatOllama(base_url=base_url, model=model)

# Системный шаблон: задаёт роль модели (учитель с определённым стилем)
system = SystemMessagePromptTemplate.from_template(
    'You are {school} teacher. You answer in short sentences'
)

# --- Цепочка 1: Факты ---

# Шаблон для запроса фактов — с параметрами темы и количества пунктов
question = HumanMessagePromptTemplate.from_template(
    'tell about the {topics} in {points} points'
)

# Сообщения для цепочки: system + пользовательский шаблон
messages = [system, question]

# Формируем полный чат-промпт
template = ChatPromptTemplate(messages)

# Цепочка: шаблон → модель → парсер (возвращает строку)
fact_chain = template.pipe(llm).pipe(StrOutputParser())

# Вызываем цепочку с конкретными параметрами
fact_output = fact_chain.invoke({
    "school": "elementary",
    "topics": "history of Ireland",
    "points": 5
})

# Выводим результат по фактам
print(fact_output)


# --- Цепочка 2: Стихотворение ---

# Шаблон пользовательского сообщения — генерация стихотворения с заданным числом предложений
poem_question = HumanMessagePromptTemplate.from_template(
    'write a poem on {topics} in {sentences}'
)

# Сообщения: system + новый пользовательский шаблон
messages = [system, poem_question]

# Новый чат-промпт для стихотворения
template = ChatPromptTemplate(messages)

# Цепочка: шаблон → модель → парсер
poem_chain = template.pipe(llm).pipe(StrOutputParser())

# Вызываем цепочку для генерации стихотворения
poem_output = poem_chain.invoke({
    "school": "elementary",
    "topics": "history of Ireland",
    "sentences": 15
})

# Выводим результат — стихотворение
print(poem_output)
