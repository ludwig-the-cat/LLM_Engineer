# Импорт необходимых компонентов из LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from tools.options import base_url, model
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# RunnableLamda для обертки кода, RunnablePassthrough выдает результат без обработки
# Инициализация модели LLM через Ollama (настройки берутся из внешнего файла)
llm = ChatOllama(base_url=base_url, model=model)

# --- Пользовательские функции обработки текста ---

# Функция возвращает количество символов в строке
def char_count(text: str) -> int:
    return len(text)

# Функция возвращает количество слов в строке
def word_count(text: str) -> int:
    return len(text.split())

# --- Промпт для генерации объяснения ---

# Шаблон чата, в который будут подставлены значения input1 и input2
prompt = ChatPromptTemplate.from_template("Explain these inputs: {input1} and {input2}")

# --- Первая цепочка: генерация текста без анализа ---

# Цепочка: шаблон → LLM → парсинг ответа в строку
chain = prompt.pipe(llm).pipe(StrOutputParser())

# Вызов цепочки с двумя входными фразами
output = chain.invoke({"input1": 'This is test', "input2": 'This is another test'})

# --- Вторая цепочка: генерация и анализ ответа ---

# Расширенная цепочка:
#   - сначала выполняется генерация текста (как в chain_)
#   - затем результат одновременно передаётся:
#       • в char_count (возвращает длину текста)
#       • в word_count (возвращает количество слов)
#       • напрямую в output (сохраняем оригинальный ответ)
chain1 = (
    prompt
    .pipe(llm)
    .pipe(StrOutputParser())
    .pipe({
        'char_count': RunnableLambda(char_count),       # Подсчёт символов
        'word_count': RunnableLambda(word_count),       # Подсчёт слов
        'output': RunnablePassthrough()                 # Просто вернуть результат как есть
    })
)

# Вызов цепочки с теми же входными данными
chain1_out = chain1.invoke({"input1": 'This is test', "input2": 'This is another test'})

# Вывод результата анализа
print(chain1_out)
