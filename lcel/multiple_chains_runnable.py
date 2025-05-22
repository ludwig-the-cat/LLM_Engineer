from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from tools.options import base_url, model

# Создаём клиента модели Ollama
llm = ChatOllama(base_url=base_url, model=model)

# Шаблон пользовательского сообщения для генерации текста с параметрами topics и points
question = HumanMessagePromptTemplate.from_template(
    'tell about the {topics} in {points} points'
)

# Шаблон системного сообщения, задающий роль модели (тип учителя)
system = SystemMessagePromptTemplate.from_template(
    'You are {school} teacher. You answer in short sentences'
)

# Объединяем system и user сообщения в чат-промт
messages = [system, question]
template = ChatPromptTemplate(messages)

# Создаём цепочку: шаблон -> модель -> парсер (возвращает строку)
chain = template.pipe(llm).pipe(StrOutputParser())

# Запускаем цепочку с параметрами для генерации текста
output = chain.invoke({
    'topics': 'solar system',
    'points': 7,
    'school': 'conservative'
})

# Создаём промт для анализа текста, который был сгенерирован на предыдущем шаге
analysis_prompt = ChatPromptTemplate.from_template("""
You are analyzer. Analyze the following text: {response}
Tell me how difficult it is to understand. Answer in two sentences only.
""")

# Создаём цепочку анализа: промт -> модель -> парсер (тоже строка)
analysis_chain = analysis_prompt.pipe(llm).pipe(StrOutputParser())

# Запускаем цепочку анализа, передавая сгенерированный текст в параметр {response}
analysis_result = analysis_chain.invoke({'response': output})

# Выводим результаты
print("Generated text:\n", output)
print("\nAnalysis:\n", analysis_result)
