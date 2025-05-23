from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from tools.options import base_url, model

# Инициализация модели с указанными параметрами подключения
llm = ChatOllama(base_url=base_url, model=model)

# --- 1. Классификация отзыва по тону (позитивный / негативный) ---

# Шаблон для классификации отзыва
# Просим модель ответить только одним словом: "Positive" или "Negative"
classification_prompt = ChatPromptTemplate.from_template("""
Given the user review below, classify it as either being about `Positive` or `Negative`.
Do not respond with more than one word.

Review: {review}
Classification:""")

# Создаём цепочку: промпт → модель → парсер строки
classification_chain = classification_prompt | llm | StrOutputParser()

# --- 2. Ответы на отзывы ---

# Шаблон для ответа на положительный отзыв
positive_prompt = ChatPromptTemplate.from_template("""
You are an expert in writing replies to positive reviews.
You need to encourage the user to share their experience on social media.
Review: {review}
Answer:""")

# Цепочка для генерации ответа на положительный отзыв
positive_chain = positive_prompt | llm | StrOutputParser()

# Шаблон для ответа на негативный отзыв
negative_prompt = ChatPromptTemplate.from_template("""
You are an expert in writing replies to negative reviews.
You need first to apologize for the inconvenience caused to the user.
You need to encourage the user to share their concern on the following Email: text@example.com
Review: {review}
Answer:""")

# Цепочка для генерации ответа на негативный отзыв
negative_chain = negative_prompt | llm | StrOutputParser()

# --- 3. Маршрутизация отзыва в нужную цепочку на основе результата классификации ---

# Функция выбора цепочки ответа в зависимости от тональности отзыва
def rout(info):
    # Если классификация положительная — используем positive_chain
    if 'positive' in info['sentiment'].lower():
        return positive_chain
    # Иначе — негативная цепочка
    else:
        return negative_chain

# --- 4. Объединённая цепочка: классификация → маршрутизация → генерация ответа ---

# Комбинируем классификацию и оригинальный текст отзыва в один словарь
# {'sentiment': <результат классификации>, 'review': <текст отзыва>}
routing_chain = {"sentiment": classification_chain, 'review': lambda x: x['review']} | RunnableLambda(rout)

# --- 5. Вызов цепочки ---

# Пример отзыва
review = "I am not happy with the service. It is not good."

# Передаём отзыв в цепочку
response = routing_chain.invoke({'review': review})

# Печатаем результат: сгенерированный ответ на отзыв
print(response)
