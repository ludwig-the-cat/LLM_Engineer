from langchain_ollama import ChatOllama
from tools.options import base_url, model
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

llm = ChatOllama(base_url=base_url, model=model)
