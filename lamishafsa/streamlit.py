import os
os.environ['GRPC_DNS_RESOLVER']='native'
import google.generativeai as genai
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#declaration respense from google
model=genai.GenerativeModel('gemini-pro')

chat=model.start_chat()
        
# Initialize the API key for the generative model
# Initialize the generative model and chat session
# Reading the text from pdf page by page and storing it into various
def get_pdf_text(pdf_docs):
    pdf_docs=("./data - data.pdf")
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

#Getting the text into number of chunks as it is helpful in faster processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
#Storing the text chunks into embeddings to retrive the answer for the query outoff it
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store =FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search_with_score(user_question)

    chain = get_conversational_chain()

    # Check if answer found from document embeddings
    if docs[0][1]< 0.7:  # Suitable threshold (adjust as needed)
        # Answer found from document, process as usual
        chain = get_conversational_chain()
        docs = new_db.similarity_search(user_question)
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        print(response)
        st.write("Reply:", response["output_text"])
    else:
        # Answer not found in documents, use Gemini conversational chain
        response=chat.send_message(user_question)
        st.write("Reply:", response.text)
        
def main():
    # Set the title of the application
    st.set_page_config(page_title="🌐💻FS Boddy Bot  ")
    st.markdown("<h1 style='text-align: center; font-size:28px;'>FS Boddy Bot  🌐💻</h1>", unsafe_allow_html=True)
    st.divider()
    model=genai.GenerativeModel('gemini-pro')
    user_question = st.chat_input("Ask question...")
    if user_question:
        user_input(user_question)
    with st.sidebar:
 # Add an image to the sidebar
     st.markdown("""
<div style="display: flex;  justify-content: top;align-items: center;">
  <img src="https://em-content.zobj.net/source/microsoft-teams/363/robot_1f916.png" width="40" height="40">
  <span style="font-size: 25px;"><b>FS Boddy Bot</b></span>
</div>
""", unsafe_allow_html=True)
# Add a link to the Faculte des sciences website in the sidebar
     st.markdown("""
<div style="display: flex; justify-content: center; align-items: flex-end; height: 75vh; position: absolute;">
  <a href="https://fsciences.univ-setif.dz/" >faculte de sciences</a>
</div>
""", unsafe_allow_html=True)
     if st.button("Clear"):
        st.session_state.chat_session = model.start_chat(history=[]) # Reset chat history
     
     
if __name__ == "__main__":
    main()

