from functools import cache

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.llms import Ollama
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

if __name__ != "__main__":
    from src.constants import MONGO_COLLECTION, MONGO_DB, MONGO_INDEX, MONGO_URL


@cache
def get_mongo_client() -> MongoClient:
    return MongoClient(MONGO_URL)


@cache
def get_collection():
    client = get_mongo_client()
    collection = client[MONGO_DB]
    return collection[MONGO_COLLECTION]


@cache
def create_search_index():
    index_model = SearchIndexModel(
        {
            "fields":[
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 384,
                    "similarity": "cosine"
                }
            ]
        },
        type="vectorSearch",
        name=MONGO_INDEX,
    )

    get_collection().create_search_index(model=index_model)


@cache
def init_ollama(model:str="llama3") -> Ollama:
    return Ollama(model=model)


@cache
def get_embed_func() -> Embeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_file_into_db(file_name):
    data_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n", " ", ""]
    )
    # TODO - add logic to prevent parsing of the file, if already available
    if file_name.endswith("docx"):
        loader = Docx2txtLoader(file_path=file_name)
    else:
        loader = PyPDFLoader(file_name)

    data = loader.load()

    all_splits = data_splitter.split_documents(data)
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=all_splits,
        embedding=get_embed_func(),
        collection=get_collection(),
        index_name=MONGO_INDEX
    )

    create_search_index()

    return vector_search


def query_rag(db: MongoDBAtlasVectorSearch, query: str) -> str:
    results = db.similarity_search_with_relevance_scores(query, k=3)
    if not results or results[0][1] < 0.6:
        # TODO log warning
        print("relevance is less than 0.6")

    response = "\n\n--\n\n".join([doc.page_content for doc, _ in results if doc])

    return response


def do_rag(query: str, context: str, db: MongoDBAtlasVectorSearch):
    prompt = PromptTemplate.from_template(
        f"""You are a helpful assistant. The user has a question.
            Answer the user question based only on the context: {context}.
            Do not add any statements like Based on the provided context.
            The user question is {query}
        """
    )
    llm = init_ollama()
    chain = (
        {"context": db.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain.invoke(query)


def ask_llm(file_name:str, query: str) -> str:
    db = load_file_into_db(file_name)
    rag_response = query_rag(db, query)

    return do_rag(query, rag_response, db)


if __name__ == "__main__":
    import argparse

    MONGO_COLLECTION = "langchain"
    MONGO_DB = "langchain_db"
    MONGO_URL = "mongodb://user:pass@localhost:27018/?directConnection=true"
    MONGO_INDEX = "SEARCH_INDEX"

    print("Make sure you have mongodb running on port 27018. Use docker-compose.yml in the repo.")
    parser = argparse.ArgumentParser(description='Process input file for local rag app.')
    parser.add_argument('filename', help='file path')
    parser.add_argument('query', help='What you want to ask the llm?')
    args = parser.parse_args()

    # db = load_file_into_db(args.filename)
    print(ask_llm(args.filename, args.query))
