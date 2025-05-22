from langchain_core.prompts import (SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate,
                                    PromptTemplate,
                                    ChatPromptTemplate)
from tools.options import BaseOptions
from langchain_ollama import ChatOllama


llm = ChatOllama(base_url=BaseOptions.base_url, model=BaseOptions.model)

question = HumanMessagePromptTemplate.from_template('tell about the {topics} in {points} points')
system = SystemMessagePromptTemplate.from_template('You are {school} teacher. You answer in short sentences')

messages = [system, question]
template = ChatPromptTemplate(messages)

# Создаем цепочку шаблон -> llm
chain = template.pipe(llm) # или chain = template | llm
response = chain.invoke({'topics': 'solar system', 'points': 7, 'school': 'conservative'})

print(response.content)


