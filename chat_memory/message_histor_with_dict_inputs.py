# Импорты из LangChain и других библиотек
from langchain_community.chat_message_histories import SQLChatMessageHistory  # Хранение истории в SQLite
from langchain_ollama import OllamaLLM  # Поддержка модели Ollama
from langchain_core.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate,
                                    MessagesPlaceholder, ChatPromptTemplate)  # Работа с промптами
from langchain_core.output_parsers import StrOutputParser  # Парсинг вывода LLM в строку
from langchain_core.messages import HumanMessage, SystemMessage  # Типы сообщений
from langchain_core.runnables import RunnableWithMessageHistory  # Добавление поддержки истории

# Импорт настроек (например, base_url и model)
import tools.options


# Функция для получения истории чата по session_id
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id=session_id, connection='sqlite:///chat_history.db')


# Создание экземпляра модели Ollama с указанием URL и имени модели
llm = OllamaLLM(base_url=tools.options.base_url, model=tools.options.model)

# Шаблон системного сообщения: задаёт роль ассистента
system = SystemMessagePromptTemplate.from_template('You are helpful assistant')

# Шаблон пользовательского сообщения: динамически подставляется из переменной {input}
human = HumanMessagePromptTemplate.from_template("{input}")

# Строим структуру промпта:
# - системное сообщение
# - место для истории сообщений (history)
# - текущее пользовательское сообщение
messages = [system, MessagesPlaceholder(variable_name='history'), human]

# Создаём финальный шаблон промпта
prompt = ChatPromptTemplate(messages=messages)

# Цепочка обработки:
# prompt -> llm -> преобразование вывода в строку
chain = prompt.pipe(llm).pipe(StrOutputParser())

# Оборачиваем цепочку в историю сообщений:
# - основная цепочка (chain)
# - функция получения истории (get_session_history)
# - имя ключа входных данных: 'input'
# - имя ключа истории: 'history'
runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='history'
)


# Функция для взаимодействия с моделью через цепочку с историей
def chat_with_llm(session_id, input):
    output = runnable_with_history.invoke(
        {'input': input},  # Передаём входные данные
        config={"configurable": {"session_id": session_id}}  # Указываем ID сессии
    )
    return output


# ID пользователя для хранения истории
user_id = 'Dmitrii'

# Сообщение от пользователя (информация о себе)
about = 'I am Dmitrii. Nice to meet you.'

# Пример вызова:
# print(chat_with_llm(user_id, about))  # Можно раскомментировать, чтобы обучить модель "контекстно"
print(chat_with_llm(user_id, 'Give it to me please'))  # Запрос к модели после контекста