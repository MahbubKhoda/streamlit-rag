import streamlit as st
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

import llm_endpoint
import embedding_endpoint

st.title('Q&A with RAG')

@st.cache_resource
def load_embedding_endpoint():
    return embedding_endpoint.get_embedding_endpoint("llm-apps-blog-gpt-j-6b-endpoint-d4041480")
my_embedding_endpoint = load_embedding_endpoint()

uploaded_file = st.file_uploader("Upload file for RAG context (only txt or pdf)", type=['txt','pdf'])
if uploaded_file is not None:
    file_name = uploaded_file.name
    if file_name.split('.')[-1] == 'txt':
        st.write("it's a text file")
    elif file_name.split('.')[-1] == 'pdf':
        st.write("it's a pdf file")
        with open('temp.pdf', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader('temp.pdf')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        texts = text_splitter.split_documents(documents)
    else:
        st.write("Cannot process the context file")
    
    docsearch = FAISS.from_documents(texts, my_embedding_endpoint)

prompt_template = """{context}\n\nGiven the above context, answer the following question and do not make up any false information, if the context doesn't have necessary information for the question simply say 'I couldn't find that information':\n{question}\n\nAnswer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

question = st.text_input('Question', None)

@st.cache_resource
def load_llm_endpoint():
    return llm_endpoint.get_llm_endpoint("llm-apps-blog-flan-t5-xxl-endpoint-d4041480")

my_llm_endpoint = load_llm_endpoint()
chain = load_qa_chain(llm=my_llm_endpoint, prompt=prompt)

if question is None:
    answer = 'Ask you question'
else:
    docs = docsearch.similarity_search(question, k=3)
    answer = chain({"input_documents": docs, "question": question}, return_only_outputs=True)["output_text"]
st.write(answer)