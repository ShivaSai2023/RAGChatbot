from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


# Load environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Ensure the key was found
if not api_key:
    raise ValueError("API key not found. Check your .env file or environment variables.")


file_path = r"D:\RAGChatbot\krishna.txt"

# Check if file exists before loading
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load the file
loader = TextLoader(file_path,encoding='utf-8')  
documents = loader.load()
print("Loaded documents:", documents)



splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
)

chunks = splitter.split_documents(documents)



embedding = OpenAIEmbeddings(openai_api_key=api_key)
  # uses `text-embedding-3-small` by default


vectorstore = FAISS.from_documents(chunks, embedding)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # top 3 chunks





# Create the LLM
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    openai_api_key=api_key
)






qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True

)

while True:
    query = input("\nAsk something (type 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain({"query": query})
    print("\nAnswer:\n", result["result"])

    print("\nSources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "unknown"))




