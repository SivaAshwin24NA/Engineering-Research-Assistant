import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FREE LOCAL EMBEDDINGS
from langchain_community.embeddings import HuggingFaceEmbeddings

# VECTOR DB
from langchain_community.vectorstores import Chroma

# LOCAL LLM
from langchain_ollama import OllamaLLM

from langchain.chains import RetrievalQA


# =========================
# LOAD DOCUMENTS
# =========================

def load_documents():

    print("📚 Loading engineering documents...")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data")

    docs = []

    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, file))
            docs.extend(loader.load())

    return docs


# =========================
# SPLIT DOCUMENTS
# =========================

def split_documents(docs):

    print("✂️ Splitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250
    )

    return splitter.split_documents(docs)


# =========================
# VECTOR DATABASE
# =========================

def create_vector_store(chunks):

    print("🧠 Creating vector database...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(BASE_DIR, "vector_db")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

    print("✅ Vector database ready!")

    return vectordb


# =========================
# QA SYSTEM (REAL AI)
# =========================

def create_qa_chain(vector_db):

    print("🤖 Loading Local AI Model...")

    llm = OllamaLLM(
        model="llama3"
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    print("🔥 Engineering AI Assistant Ready!")

    return qa_chain


# =========================
# MAIN
# =========================

def main():

    docs = load_documents()

    if not docs:
        raise ValueError("❌ No PDFs found inside /data")

    chunks = split_documents(docs)

    vector_db = create_vector_store(chunks)

    qa_chain = create_qa_chain(vector_db)

    print("\nType 'exit' anytime.\n")

    while True:

        question = input("Ask engineering question: ")

        if question.lower() == "exit":
            break

        response = qa_chain.invoke({"query": question})

        print("\n📘 Answer:\n")
        print(response["result"])
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
