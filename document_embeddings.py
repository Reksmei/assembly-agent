from langchain_chroma import Chroma 
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pathlib
import os
from dotenv import load_dotenv

# Load environmental credential to access EmbeddingGemma
load_dotenv()

os.getenv("HF_TOKEN")

def main():

    load_dotenv()

    os.getenv("GEMINI_API_KEY")

    # Load the PDF Instruction Manual
    loader = PyPDFLoader(pathlib.Path(__file__).parent / "brimes-assembly.pdf")
    pages = loader.load()

    # Split the text into managable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n"]
    )

    docs = text_splitter.split_documents(pages)

    # Initialize the Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="google/embeddinggemma-300m",
    )

    # Store and persist the chunks in local ChromaDB instance
    vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./vectorDB"
    )

    # Convert vector store to a retriever for querying
    retrieval = vectorstore.as_retriever(search_kwargs={"k": 5})

# Make the script executable
if __name__ == "__main__":
    print("Starting Document Embedding Script")
    main()
    

