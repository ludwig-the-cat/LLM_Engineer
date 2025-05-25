# Импортируем Ollama-модель и параметры подключения
from langchain_ollama import ChatOllama
from tools.options import base_url, model

# Импорт шаблонов сообщений и Pydantic парсера
from langchain_core.prompts import (
    PromptTemplate
)
from langchain_core.output_parsers import PydanticOutputParser

# Импорт схемы для структурированного вывода (ты должен создать ее в tools/schemas.py)
from tools.schemas import JokeOutputSchema

# Инициализируем LLM
llm = ChatOllama(base_url=base_url, model=model)

# Создаем Pydantic парсер, который будет использовать схему JokeOutputSchema
parser_out = PydanticOutputParser(pydantic_object=JokeOutputSchema)

# Получаем текстовую инструкцию по форматированию, которую вставим в промпт
instruction = parser_out.get_format_instructions()

# Создаем шаблон запроса, передавая инструкцию по формату как "partial"
prompt = PromptTemplate(
    template="""
    Answer the user query with a joke. Here is your formatting instruction.
    {format_instruction}

    Query: {query}

    Answer:
    """,
    input_variables=['query'],
    partial_variables={'format_instruction': instruction}
)

# Создаем первую цепочку: шаблон → LLM
# Эта цепочка выводит обычный текст (без структуры)
chain = prompt.pipe(llm)

# Пример вызова: получаем обычный текст от модели
output = chain.invoke({'query': 'Tell me a joke about cats'})
print("Raw LLM Output:\n", output.content)

# Создаем вторую цепочку: шаблон → LLM → парсер
# Эта цепочка уже возвращает структурированные данные по Pydantic-схеме
chain_parser = prompt.pipe(llm).pipe(parser_out)

# Пример вызова: результат будет объектом `JokeOutputSchema`
parse_out = chain_parser.invoke({'query': 'Tell me a joke about dogs'})
print("\nParsed Output:\n", parse_out)
