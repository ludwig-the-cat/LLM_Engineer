from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from tools.options import base_url, model
from langchain_core.output_parsers import JsonOutputParser
from tools.schemas import JokeOutputSchema

llm = ChatOllama(base_url=base_url, model=model)
parser = JsonOutputParser(pydantic_object=JokeOutputSchema)

prompt = PromptTemplate(
    template="""
    Answer the user query with a joke. Here is your formatting instruction.
    {format_instruction}

    Query: {query}

    Answer:
    """,
    input_variables=['query'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = prompt | llm | parser
response = chain.invoke({'query': 'Tell me a joke about cat'})
print(response)
