from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from tools.options import base_url, model
from langchain.output_parsers import DatetimeOutputParser

llm = ChatOllama(base_url=base_url, model=model)
parser = DatetimeOutputParser()
instruction = parser.get_format_instructions()
prompt = PromptTemplate(
    template="""
    Tell me when {event} was happened. Answer only with a datetime.
    Formating: {format_instruction}
    """,
    input_variables=['event'],
    partial_variables={'format_instruction': instruction}
)

chain = prompt.pipe(llm).pipe(parser)
output = chain.invoke({'event': 'First step on the Moon'})
print(output)
