from langchain_ollama import ChatOllama
from tools.options import base_url, model
from tools.schemas import JokeOutputSchema

llm = ChatOllama(base_url=base_url, model=model)
structured_llm = llm.with_structured_output(JokeOutputSchema)
output = structured_llm.invoke('Tell me joke about cat')
print(output)
