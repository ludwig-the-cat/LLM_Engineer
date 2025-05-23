# Импорт необходимых компонентов из LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from tools.options import base_url, model
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables import chain

llm = ChatOllama(base_url=base_url, model=model)

@chain
def custom_chain(params):
    return {
        'fact': fact_chain.invoke(params),
        'poem': poem_chain.invoke(params),
    }

params = {'school': 'a', 'topics': 'lunar system', 'points': 2, 'sentences': 2}
output = custom_chain.invoke(params)
print(output['poem'])
print()
print(output['fact'])