# Импорт всех необходимых компонентов
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from tools.options import base_url, model
from langchain_core.runnables import chain  # <-- используем декоратор @chain

# Инициализация LLM
llm = ChatOllama(base_url=base_url, model=model)

# --- Создание шаблона и цепочки для ФАКТОВ ---
fact_system_template = "You are {school} teacher. You answer in short sentences."
fact_user_template = "Tell about the {topics} in {points} points."

fact_prompt = ChatPromptTemplate.from_messages([
    ("system", fact_system_template),
    ("human", fact_user_template)
])

fact_chain = fact_prompt | llm | StrOutputParser()

# --- Создание шаблона и цепочки для СТИХА ---
poem_user_template = "Write a poem on {topics} in {sentences} sentences."

poem_prompt = ChatPromptTemplate.from_messages([
    ("system", fact_system_template),  # та же роль учителя
    ("human", poem_user_template)
])

poem_chain = poem_prompt | llm | StrOutputParser()

# --- Кастомная цепочка с декоратором ---
# Эта функция будет вызвана как LangChain Runnable, но внутри использует Python-логику
@chain
def custom_chain(params: dict) -> dict:
    return {
        'fact': fact_chain.invoke(params),
        'poem': poem_chain.invoke(params),
    }

# --- Пример использования ---
params = {
    'school': 'a',
    'topics': 'lunar system',
    'points': 2,
    'sentences': 2
}

result = custom_chain.invoke(params)

# --- Вывод результата ---
print("Poem:\n", result['poem'])
print("\nFacts:\n", result['fact'])
