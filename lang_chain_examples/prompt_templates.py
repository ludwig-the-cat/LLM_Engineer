from pydoc_data.topics import topics

from langchain_core.prompts import (SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate,
                                    PromptTemplate,
                                    ChatPromptTemplate)
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from tools.options import BaseOptions

llm = ChatOllama(base_url=BaseOptions.base_url, model=BaseOptions.model)

# Формируем шаблоны системного и пользовательского сообщения
question = HumanMessagePromptTemplate.from_template('tell about the {topics} in {points} points')
system = SystemMessagePromptTemplate.from_template('You are {school} teacher. You answer in short sentences')

# Формируем запрос
messages = [system, question]

# Формируем шаблон запроса в ollama
template = ChatPromptTemplate(messages)

full_question = template.invoke({'school': 'grim', 'topics': 'night', 'points': 5})
response = llm.invoke(full_question)
print(response.content)
