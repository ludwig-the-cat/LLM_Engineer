# Импорт необходимых модулей из библиотек LangChain
from langchain_core.output_parsers import StrOutputParser  # Для преобразования вывода модели в строку
from langchain_core.prompts import ChatPromptTemplate      # Для создания шаблонов промптов
from langchain_ollama import ChatOllama                    # Для работы с моделью Ollama
from tools.options import base_url, model                  # Предположительно, это настройки для подключения к Ollama
from langchain_core.runnables.history import RunnableWithMessageHistory  # Для добавления истории сообщений
from langchain_community.chat_message_histories import SQLChatMessageHistory  # Для хранения истории чата в SQLite


# Функция, возвращающая историю сообщений для конкретной сессии
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id=session_id, connection='sqlite:///chat_history.db')


# Создание экземпляра модели ChatOllama с указанием базового URL и названия модели
llm = ChatOllama(base_url=base_url, model=model)

# Создание шаблона промпта — простой плейсхолдер {prompt}, который будет заменён при вызове
template = ChatPromptTemplate.from_template("{prompt}")

# Цепочка обработки: шаблон -> модель -> парсер строки
chain = template.pipe(llm).pipe(StrOutputParser())

# Текст о пользователе (или персоне), который мы передадим модели
about = 'I am Ludwig the Cat. I have a lot of fur'

# Пример того, как можно вызвать цепочку напрямую (закомментирован)
# response = chain.invoke({'prompt': about})
# print(response)

# Текст запроса — вопрос о имени
prompt = 'What is my name?'

# Оборачиваем цепочку в поддержку истории сообщений
runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history  # Указываем функцию для получения истории по session_id
)

# Уникальный идентификатор пользователя
user_id = 'Ludwig'

# Получаем историю сообщений для этого пользователя
history = get_session_history(user_id)

# Очистка истории (если нужно начать с чистого листа)
# history.clear()

# Вызов цепочки с историей
response = runnable_with_history.invoke(
    {"prompt": about},  # Передаём контекст или информацию
    config={"configurable": {"session_id": user_id}}  # Указываем ID сессии
)

# Вывод ответа модели
print(response)

# Очистка истории после выполнения (для тестирования или повторного запуска)
history.clear()