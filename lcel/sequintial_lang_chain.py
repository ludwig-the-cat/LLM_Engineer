# Импорт шаблонов для системного и пользовательского сообщений + общего шаблона чата
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate
)

# Импорт настроек: адрес Ollama-сервера и название модели
from tools.options import base_url, model

# Импорт клиента Ollama, обёртки над языковой моделью
from langchain_ollama import ChatOllama

# Создаём объект языковой модели с подключением к Ollama
llm = ChatOllama(base_url=base_url, model=model)

# Шаблон пользовательского сообщения, где можно подставлять темы и количество пунктов
question = HumanMessagePromptTemplate.from_template(
    'tell about the {topics} in {points} points'
)

# Шаблон системного сообщения: указывает роль LLM — в данном случае "учитель" с определённым стилем
system = SystemMessagePromptTemplate.from_template(
    'You are {school} teacher. You answer in short sentences'
)

# Объединяем оба шаблона в список сообщений
messages = [system, question]

# Создаём общий шаблон диалога, состоящий из system и human сообщений
template = ChatPromptTemplate(messages)

# Создаём цепочку: шаблон -> языковая модель
# То же самое, что `template | llm`, просто в более читаемом виде
chain = template.pipe(llm)

# Подставляем значения в шаблон и запускаем цепочку: происходит генерация промта + вызов модели
response = chain.invoke({
    'topics': 'solar system',    # тема
    'points': 7,                 # количество пунктов
    'school': 'conservative'     # стиль ответа
})

# Печатаем содержимое ответа (AIMessage.content)
print(response.content)
