# main.py

#####################################################################
# Amazon Bedrock - boto3
#####################################################################

import boto3
import os
from dotenv import load_dotenv

load_dotenv()

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.environ["AWS_DEFAULT_REGION"],
)

#####################################################################
# LLM - Amazon Bedrock LLM using LangChain
#####################################################################

from llama_index.llms import LangChainLLM
from langchain.llms import Bedrock

model_id = "anthropic.claude-v2"
model_kwargs =  { 
    "max_tokens_to_sample": 4096,
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

llm = Bedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs
)

#####################################################################
# Embedding Model - Amazon Titan Embeddings Model using LangChain
#####################################################################

from llama_index import LangchainEmbedding, StorageContext, load_index_from_storage
from langchain.embeddings import BedrockEmbeddings

# create embeddings
bedrock_embedding = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id="amazon.titan-embed-text-v1",
)

# load in Bedrock embedding model from langchain
embed_model = LangchainEmbedding(bedrock_embedding)

#####################################################################
# Service Context
#####################################################################

from llama_index import ServiceContext, set_global_service_context

service_context = ServiceContext.from_defaults(
  llm=llm,
  embed_model=embed_model,
  system_prompt="You are an AI assistant answering questions."
)

set_global_service_context(service_context)

#####################################################################
# Streamlit
#####################################################################

import streamlit as st
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex

st.set_page_config(
  page_title="LlamaIndex Q&A over you data 📂",
  page_icon="🦙",
  layout="centered",
  initial_sidebar_state="auto",
  menu_items=None)

st.title("LlamaIndex 🦙 Q&A over your data 📂")

@st.cache_resource(show_spinner=False)
def load_data():
  """
    Loads and indexes the data using the VectorStoreIndex.
    
    Returns:
    - VectorStoreIndex: Indexed representation of your data.
  """
  # with st.spinner(
  #   text="Loading and indexing your data. This may take a while..."):
  #   reader=SimpleDirectoryReader(input_dir="./data", recursive=True)
  #   docs=reader.load_data()

  #   index=VectorStoreIndex.from_documents(docs)
  #   return index

# Create Index
# index=load_data()

storage_context = StorageContext.from_defaults(persist_dir="./storage/")
index = load_index_from_storage(storage_context, service_context=service_context)

# Create Query Engine
query_engine=index.as_query_engine(similarity_top_k=3)

# index.storage_context.persist()

# Take input from the user
user_input=st.text_input("Enter Your Query", "")

# Display the input
if st.button("Submit"):
  st.write(f"Your Query: {user_input}")

  with st.spinner("Thinking..."):
    # Query the index
    result=query_engine.query(f"\n\nHuman:{user_input}\n\nAssistant:")

    # Display the results
    st.write(f"Answer: {str(result)}")
    