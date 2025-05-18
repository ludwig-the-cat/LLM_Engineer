import ollama

from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


doc_path = "./data/The Ultimate QA Testing Handbook.pdf"
model = "llama3.2:1b"

if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    loader_data = loader.load()
    print('Закончил загрузку')
else:
    print("Upload a PDF file")

# Посмотреть что там в после загрузки pdf
#content = loader_data[0].page_content

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
chunks = text_splitter.split_documents(loader_data)
print('Закончил разделение', len(chunks))

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text:latest"),
    collection_name="simple_rag"
)
print('Закончил создавать векторную БД')

llm = ChatOllama(model=model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are QA guru. Your task generate three different versions of the given user question
    to retrieve relevant answers from a vector database. Provide these alternative questions separated by newlines
    Original question: {question}.""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based on the following context: 
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = chain.invoke(input=("Give me a mention of Unit testing",))
print(res)