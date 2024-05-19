from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
from openai import OpenAI

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
        
# Initialize the OpenAI client with the API key
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

os.environ["OPENAI_API_KEY"] = "xxx"

# initializing the embeddings
embeddings = OpenAIEmbeddings()

# default model = "gpt-3.5-turbo"
llm = ChatOpenAI()

directory = "./nostradamos"

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

documents = load_docs(directory)

def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

db = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings
)

chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
    similar_docs = db.similarity_search(query, k=2) # get 3 closest chunks
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

def mistral7b(user_message, system_message):
    
    # Create a chat completion request
    completion = client.chat.completions.create(
        model="local model",  # Model is currently unused but required for the function call
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
    )
    # Return just the content of the generated message
    return completion.choices[0].message.content  # Adjusted this line
    
system_message = open_file("chatbot1.txt")
    
while True:
    prompt = input(f"{YELLOW}Enter your query here ('exit' to leave): {RESET_COLOR}")
    # Break the loop if a certain condition is met, e.g., if the prompt is 'exit'
    if prompt.lower() == 'exit':
        break

    answer = get_answer(prompt)
    answer2 = f"{CYAN}Context: {answer} \n User Query: {prompt}{RESET_COLOR}"
    print(answer2)

    chatbot_response = mistral7b(answer2, system_message)
    print(NEON_GREEN + chatbot_response + RESET_COLOR)
   
