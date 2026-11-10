from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
from dotenv import load_dotenv
import os

# Load environmental credential to access EmbeddingGemma
load_dotenv()

os.getenv("HF_TOKEN")

# Initialize constants for embedding user queries
embeddings = HuggingFaceEmbeddings(
    model_name="google/embeddinggemma-300m",
)

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./vectorDB"
)

@tool
def retrieve_relevant_info(query_str: str):
    '''
    Retrieves the relevant chunks from the instruction manual to answer the user's questions.

    Args:
      query_str(str): The string that will be used to query the vector database. E.g. Tool to attach drawer back panel
    '''
    
    # Query the Vector DB for similar docs to the input query
    results = vectorstore.similarity_search(query_str, k=3)
    
    # Return the document chunks for agent to use
    return [f"{idx+1} (Page {doc.metadata.get('page', 'N/A')}): {doc.page_content}" for idx, doc in enumerate(results)]

