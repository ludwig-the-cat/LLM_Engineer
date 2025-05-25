# Импорт промпт-шаблона PromptTemplate
from langchain_core.prompts import PromptTemplate

# Импорт LLM клиента для Ollama
from langchain_ollama import ChatOllama

# Импорт настроек подключения: base_url и model
from tools.options import base_url, model

# Импорт парсера, который разбирает вывод модели в список, разделённый запятыми
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# Инициализация модели ChatOllama с параметрами
llm = ChatOllama(base_url=base_url, model=model)

# Создаем парсер, который будет ожидать список значений, разделённых запятыми
parser = CommaSeparatedListOutputParser()

# Получаем строку-инструкцию по формату для включения в prompt
format_instruction = parser.get_format_instructions()

# Создаем шаблон промпта, включающий инструкцию и переменную запроса
prompt = PromptTemplate(
    template="""
    Answer the user query with a list of values.
    Here is your format instruction: {format_instruction}

    Query: {query}

    Answer:
    """,
    input_variables=['query'],  # переменные, которые мы передаем при вызове
    partial_variables={'format_instruction': format_instruction}  # вставляется автоматически
)

# Строим цепочку: prompt → модель → парсер
# Модель генерирует текст, парсер превращает его в список значений
chain = prompt | llm | parser

# Пример вызова цепочки: запросим 15 ключевых SEO слов для сайта о масштабных моделях
output = chain.invoke({
    'query': 'Tell me 15 SEO keywords. I have content about scale models'
})

# Выводим полученный список
print(output)
