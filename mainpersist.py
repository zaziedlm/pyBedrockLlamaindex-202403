# main.py

#####################################################################
# Amazon Bedrock - boto3
#####################################################################

import boto3
import os
#from dotenv import load_dotenv

#load_dotenv()

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    # region_name=os.environ["AWS_DEFAULT_REGION"],
)

#####################################################################
# LLM - Amazon Bedrock LLM using LangChain
#####################################################################

from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.bedrock import Bedrock

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
    model=model_id,
    #kwargs=model_kwargs,
    # aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    # aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    # #aws_session_token="AWS Session Token to use",
    # region_name=os.environ["AWS_DEFAULT_REGION"],
)

#####################################################################
# Embedding Model - Amazon Titan Embeddings Model using LangChain
#####################################################################

from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.bedrock import BedrockEmbedding

# create embeddings
embed_model = BedrockEmbedding(
    client=bedrock_runtime,
    model="amazon.titan-embed-text-v1",
    #model=Models.TITAN_ENBEDDING,  # default?
    # aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    # aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    # # #aws_session_token="AWS Session Token to use",
    # region_name=os.environ["AWS_DEFAULT_REGION"],
)

#####################################################################
# Service Context
#####################################################################

# from llama_index import ServiceContext, set_global_service_context

# service_context = ServiceContext.from_defaults(
#   llm=llm,
#   embed_model=embed_model,
#   system_prompt="You are an AI assistant answering questions."
# )

# set_global_service_context(service_context)

# ServiceContext was deprecated, globally in the Settings
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
Settings.num_output = 256
Settings.context_window = 3900
#####################################################################
# Streamlit
#####################################################################

import streamlit as st
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

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
index = load_index_from_storage(storage_context)

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
    