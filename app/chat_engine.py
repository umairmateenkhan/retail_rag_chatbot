
from langchain.agents import initialize_agent, Tool
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

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
support_loader = CSVLoader(file_path="data/Retail_Support_Singleline.csv",
                           source_column="question",
                           encoding="utf-8")
support_doc = support_loader.load()

# Load inventory dataset
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
# chunked_support = split_by_lines(support_doc, lines_per_chunk=10)     
chunked_inventory = split_by_lines(inventory_doc, lines_per_chunk=10)

# Initialize the LLM
# Using gpt-4o for better performance
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)


# Vector DB for support and inventory   
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
# Create vector stores and retrievers for support
vector_db_support = FAISS.from_documents(support_doc, embeddings)
retriever_support = vector_db_support.as_retriever()
support_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever_support,
    chain_type="stuff",# combine the retrieved documents into a single prompt that is passed to the LLM.
    return_source_documents=True
)
# Create vector stores and retrievers for inventory
vector_db_inventory = FAISS.from_documents(chunked_inventory, embeddings)           
retriever_inventory = vector_db_inventory.as_retriever()
inventory_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever_inventory,
    chain_type="stuff",# combine the retrieved documents into a single prompt that is passed to the LLM.
    return_source_documents=True
)



# Function to check order status
def check_order_status(text: str) -> str:
    for order_id in orders_df['order_id']:
        if order_id.lower() in text.lower():
            row = orders_df[orders_df['order_id'] == order_id].iloc[0]
            return f"Order {order_id} for {row['product']} is currently '{row['status']}' and shipping to {row['delivery_city']}"
    return "Order ID not found."

# Create memory buffer
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Initialize the agent with tools
tools = [
    Tool(
        name="Check Order Status",
        func=check_order_status,
        description=(
            "Use this tool to check the status of an order by providing the order ID."
            "Input should be a natural language query like 'What is the status of order 12345?' or 'Can you tell me about order 67890?' or 'Where is my order 12345?'."
        )
    ),
    Tool(
        name="Support Information",
        func=lambda q: support_chain.invoke(q)["result"],
        description=(
            "Use this tool to retrieve support information related to Installation,Managed IT Services,Warranty,Wholesale, returns, delivery, refunds. It searches the official support documentation."
            "Input should be a natural language query like 'What is the return policy?' or  'What is the warranty policy?' or 'Do you offer bulk pricing?' or 'Does MacBook Air support dual monitors?' or 'What are your delivery charges' or 'Can I outsource my IT support to you?' or 'Do you offer installation for MacBook Air?'."
        )
    ),
    Tool(
        name="Inventory Information",
        func=lambda q: inventory_chain.invoke(q)["result"],
        description=(
            "Use this tool to check product availability in the inventory. "
            "Input should be a product name or natural language query like "
            "'Is Raspberry Pi 4 available?' or 'Do you have Asus ROG Gaming Laptop? or 'What is the stock for iPhone 14 Pro Max?' or 'How many iPhone 14 Pro Max are available in Auckland?' or 'How many iPhone 14 Pro Max are available in Wellington?' or 'How many iPhone 14 Pro Max are available in Christchurch?' or 'How many iPhone 14 Pro Max are available in Hamilton?'. or 'Do you sell iPhone 14 Pro Max?'."
        )
    )
]





agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="conversational-react-description",
    memory=memory,
    handle_parsing_errors=True,
    verbose=False,
     agent_kwargs={
         "system_message": (
            "You are a helpful retail assistant that helps users with inventory, product questions, and support issues. "
            "You can use tools like 'Inventory Information' or 'Support Lookup' to provide accurate responses. "
            "Use a tool if it can help you answer better. Only respond directly if you're confident and no tool is needed.\n\n"

            "If the question is unclear or lacks detail, ask a follow-up question before answering or using a tool.\n\n"

            "Respond using **only one** of the following formats:\n"
            "- To show your reasoning:\n"
            "  Thought: [your thought process]\n"
            "- If using a tool:\n"
            "  Thought: [your reasoning]\n"
            "  Action: [tool name]\n"
            "  Action Input: [what you're sending to the tool]\n"
            "- If giving a final answer without a tool:\n"
            "  Final Answer: [your complete response to the user]\n\n"

            "⚠️ Do not mix formats. Do not output both Action and Final Answer together.\n"
            "❗ If you're unsure, ask for clarification instead of guessing.\n"
            "✅ Always be helpful, polite, and clear in your responses."
         )
     }
)


# The agent is now ready to handle queries related to order status, support, and inventory.
# Function to get chat response
# This function runs the agent with the user's query and returns the response   
def get_chat_response(query: str) -> str:
    return agent.run(query)

# # Main loop for user interaction
# if __name__ == "__main__":
#     print("Retail RAG Chat Agent. Type 'exit' to quit.")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             break
#         response = get_chat_response(user_input)
#         print("Agent:", response)