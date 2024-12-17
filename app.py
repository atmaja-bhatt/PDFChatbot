import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pickle

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#go through all the pages and extract the text
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

#after getting all the extracted text, divide them in smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#converting the chunks into vectors
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index") #shows that the vectors will be stored locally under the name "faiss_index"


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

#load gemini model -> create a template -> get the chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

# give the quest -> similarity search on all faiss index/search created -> 
# create the conversational chain-> 
# in response we give the chain and the resp


def main():
    st.set_page_config("Chat PDF")
    st.header("PDF CHATBOT")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()























# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import pickle

# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Go through all the pages and extract the text
# def get_pdf_text(pdf_docs):
#     pdf_texts = {}
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#         pdf_texts[pdf.name] = text
#     return pdf_texts

# # After getting all the extracted text, divide them in smaller chunks
# def get_text_chunks(pdf_texts):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = []
#     for pdf_name, text in pdf_texts.items():
#         pdf_chunks = text_splitter.split_text(text)
#         for chunk in pdf_chunks:
#             chunks.append((pdf_name, chunk))
#     return chunks

# # Converting the chunks into vectors
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     texts = [chunk[1] for chunk in text_chunks]
#     vector_store = FAISS.from_texts(texts, embedding=embeddings)

#     for i, chunk in enumerate(text_chunks):
#         vector_store.add_metadata(i, {"pdf_name": chunk[0]})

#     vector_store.save_local("faiss_index") # shows that the vectors will be stored locally under the name "faiss_index"

#     with open("chunk_metadata.pkl", "wb") as f:
#         pickle.dump(text_chunks, f)

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)

#     with open("chunk_metadata.pkl", "rb") as f:
#         chunk_metadata = pickle.load(f)

#     relevant_chunks = [chunk_metadata[doc.metadata['index']] for doc in docs]

#     context_with_sources = "\n\n".join([f"From {chunk[0]}: {chunk[1]}" for chunk in relevant_chunks])
    
#     chain = get_conversational_chain()

#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

#     st.write("Reply: ", response["output_text"])
#     st.write("Sources: ", context_with_sources)

# def main():
#     st.set_page_config(page_title="Chat PDF")
#     st.header("PDF CHATBOT")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
#         if st.button("Submit"):
#             with st.spinner("Processing..."):
#                 pdf_texts = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(pdf_texts)
#                 get_vector_store(text_chunks)
#                 st.success("Done")

# if __name__ == "__main__":
#     main()

    
    
    

