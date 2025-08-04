
from langchain.agents import initialize_agent, Tool
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader

import pandas as pd
from langchain.schema import Document
import openai
import os
from dotenv import load_dotenv


load_dotenv(override=True)


# Ensure the OpenAI API key is set
openai.api_key = os.environ["OPENAI_API_KEY"]


# Load datasets
orders_df = pd.read_csv("data/Retail_Order_Status_Dataset.csv")
support_loader = CSVLoader(file_path="data/Realistic_Retail_Support_Dataset.csv")
support_doc = support_loader.load()
inventory_loader = CSVLoader(file_path="data/Retail_Inventory_Availability_Dataset.csv")
inventory_doc = inventory_loader.load()


# Split documents based on lines (5 to 10 lines per chunk)
def split_by_lines(documents, lines_per_chunk=7):
    chunks = []
    for doc in documents:
        lines = doc.page_content.splitlines()
        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i:i + lines_per_chunk]
            chunk_text = "\n".join(chunk_lines)
            chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))
    return chunks

# Apply line-based chunking
chunked_support = split_by_lines(support_doc, lines_per_chunk=10)     
chunked_inventory = split_by_lines(inventory_doc, lines_per_chunk=10)

# Vector DB for support and inventory   
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
# Create vector stores and retrievers for support
vector_db_support = FAISS.from_documents(chunked_support, embeddings)
retriever_support = vector_db_support.as_retriever()
support_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever_support)
# Create vector stores and retrievers for inventory
vector_db_inventory = FAISS.from_documents(chunked_inventory, embeddings)           
retriever_inventory = vector_db_inventory.as_retriever()
inventory_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever_inventory)

# Function to check order status
def check_order_status(text: str) -> str:
    for order_id in orders_df['order_id']:
        if order_id.lower() in text.lower():
            row = orders_df[orders_df['order_id'] == order_id].iloc[0]
            return f"Order {order_id} for {row['product']} is currently '{row['status']}' and shipping to {row['delivery_city']}"
    return "Order ID not found."

# Initialize the agent with tools
tools = [
    Tool(
        name="Check Order Status",
        func=check_order_status,
        description="Use this tool to check the status of an order by providing the order ID."
    ),
    Tool(
        name="Support Information",
        func=support_chain.run,
        description="Use this tool to retrieve support information related to retail operations."
    ),
    Tool(
        name="Inventory Information",
        func=inventory_chain.run,
        description="Use this tool to retrieve inventory availability information."
    )
]

# The agent is set up to use zero-shot reasoning with the provided tools    
agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(),
    agent="zero-shot-react-description",
    verbose=False
)

# The agent is now ready to handle queries related to order status, support, and inventory.
# Function to get chat response
# This function runs the agent with the user's query and returns the response   
def get_chat_response(query: str) -> str:
    return agent.run(query)

# Main loop for user interaction
if __name__ == "__main__":
    print("Retail RAG Chat Agent. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = get_chat_response(user_input)
        print("Agent:", response)