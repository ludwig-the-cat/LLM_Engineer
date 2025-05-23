from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_core.runnables import RunnableParallel
from langchain_ollama import ChatOllama
from tools.options import base_url, model

# Инициализация модели Ollama с настройками
llm = ChatOllama(base_url=base_url, model=model)

# Системный шаблон — задаёт стиль общения
system = SystemMessagePromptTemplate.from_template(
    'You are {school} teacher. You answer in short sentences.'
)

# --- Цепочка 1: Генерация фактов ---

# Шаблон запроса для фактов
fact_question = HumanMessagePromptTemplate.from_template(
    'Tell about the {topics} in {points} points.'
)

# Формируем промпт
fact_prompt = ChatPromptTemplate.from_messages([system, fact_question])

# Создаём цепочку: промпт → модель → парсер строки
fact_chain = fact_prompt | llm | StrOutputParser()


# --- Цепочка 2: Генерация стихотворения ---

# Шаблон запроса для стихотворения
poem_question = HumanMessagePromptTemplate.from_template(
    'Write a poem on {topics} in {sentences} sentences.'
)

# Формируем промпт
poem_prompt = ChatPromptTemplate.from_messages([system, poem_question])

# Создаём цепочку: промпт → модель → парсер строки
poem_chain = poem_prompt | llm | StrOutputParser()


# --- Параллельное выполнение обеих цепочек ---

# Объединяем цепочки в параллельный исполняемый блок
combined_chain = RunnableParallel(fact=fact_chain, poem=poem_chain)

# Выполняем обе цепочки с заданными параметрами
result = combined_chain.invoke({
    'school': 'elementary',
    'topics': 'History of Iceland',
    'points': 5,
    'sentences': 15
})

# Вывод результата
print("FACTS:\n")
print(result['fact'])
print("\nPOEM:\n")
print(result['poem'])
