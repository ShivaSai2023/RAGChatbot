from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Check your .env file or environment variables.")

# List of file paths
file_paths = [
    r"D:\RAGChatbot\krishna.txt",
    r"D:\RAGChatbot\shivasai.txt"
]

# Load and combine documents
all_documents = []

for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        continue
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        all_documents.extend(documents)
    except Exception as e:
        print(f"⚠️ Error loading {file_path}: {e}")

if not all_documents:
    raise RuntimeError("No documents were loaded.")

print(f"✅ Loaded {len(all_documents)} documents.")

# Split text into chunks
splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
)
chunks = splitter.split_documents(all_documents)

# Create embeddings
embedding = OpenAIEmbeddings(openai_api_key=api_key)

# Build vector store
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# Setup LLM
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    openai_api_key=api_key
)

# Setup QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Chat loop
while True:
    query = input("\nAsk something (type 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain({"query": query})
    print("\nAnswer:\n", result["result"])

    print("\nSources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "unknown"))
