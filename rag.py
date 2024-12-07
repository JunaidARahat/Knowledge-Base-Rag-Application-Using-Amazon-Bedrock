import os
import boto3
import streamlit as st
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.llms.bedrock import Bedrock
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")
region_name = os.getenv("region_name")

# Define the prompt template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but summarize with 
at least 250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

# Initialize Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# Get embedding model
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", client=bedrock
)

# Function to load and process documents
def get_documents():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to create and save FAISS vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# Function to initialize LLM
def get_llm():
    llm = Bedrock(
        model_id="meta.llama3-70b-instruct-v1:0",
        client=bedrock,
        model_kwargs={"max_gen_len": 512},
    )
    return llm

# Define prompt template for RetrievalQA
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Function to get response from the LLM
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    answer = qa({"query": query})
    return answer["result"]

# Main application
def main():
    st.set_page_config(page_title="RAG Demo")
    st.header("End-to-End RAG Application")
    user_question = st.text_input("Ask a question from the PDF files:")

    with st.sidebar:
        st.title("Update or Create Vector Store")
        if st.button("Store Vector"):
            with st.spinner("Processing..."):
                docs = get_documents()
                get_vector_store(docs)
                st.success("Vector store created successfully!")

    if st.button("Send"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(
                "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True
            )
            llm = get_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))

# Run the application
if __name__ == "__main__":
    main()
