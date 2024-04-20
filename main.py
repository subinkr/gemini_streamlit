import streamlit as st
import langchain
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import os
import json

load_dotenv()

from datetime import datetime

NO_DATA = "데이터가 없습니다."

def get_json_data():
    loader = JSONLoader(file_path="data/counsel_data.json", jq_schema='.[].talk', text_content=False)
    documents = loader.load()

    text = ''
    for i, doc in enumerate(documents):
        content_keys = json.loads(documents[i].page_content)["content"].keys()
        temp = ''
        for key in content_keys:
            temp += json.loads(documents[i].page_content)["content"][key]
        text += temp + '\n'

    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key=None):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(vectorstore=None, api_key=None):
    prompt_template = """
    너는 이제 카운슬러야. 아래 제공된 데이터와 이전 상담 내용을 기반으로 질문자의 감정에 공감해주고 질문자의 기분이 나아질만한 답변을 생성하는데 답변 내용은 항상 3줄 이내로 요약해서 이야기해줘.\n
    데이터 안에 해당하는 내용이 없으면 "{no_data}"로 대답해줘.\n\n
    데이터: \n{context}\n
    이전 상담 내용: \n{history}\n
    질문: \n{question}\n

    답변: 
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "history", "no_data"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key, conversation_history):
    data = get_json_data()
    text_chunks = get_text_chunks(data)
    vector_store = get_vector_store(text_chunks, api_key)
    user_question_output = ""
    response_output = ""
    
    question_history = ''
    user_question_output = user_question

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    for conversation in conversation_history:
        question_history += conversation[0] + ' ' + conversation[1] + ' '

    chain = get_conversational_chain(vectorstore=new_db, api_key=api_key)
    response = chain({"input_documents": docs, "question": user_question, "no_data": NO_DATA, "history": question_history}, return_only_outputs=True)
    response_output = response['output_text']
    conversation_history.append((user_question_output, response_output, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", "))
    st.session_state.user_question = ""  # Clear user question input 

    st.markdown(
        f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{
                background-color: #2b313e;
            }}
            .chat-message.bot {{
                background-color: #475063;
            }}
            .chat-message .avatar {{
                width: 20%;
            }}
            .chat-message .avatar img {{
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .chat-message .message {{
                width: 80%;
                padding: 0 1.5rem;
                color: #fff;
            }}
            .chat-message .info {{
                font-size: 0.8rem;
                margin-top: 0.5rem;
                color: #ccc;
            }}
        </style>
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>    
            <div class="message">{user_question_output}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
            </div>
            <div class="message">{response_output}</div>
            </div>
            
        """,
        unsafe_allow_html=True
    )

    # Show history
    for question, answer, timestamp, pdf_name in reversed(conversation_history[:-1]):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>    
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    st.set_page_config(page_title="AI Counselor", page_icon=":robot-face:")
    st.header("안녕하세요 AI 상담사입니다. :flag-kr:")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    api_key = "YOUR_API_KEY"

    st.session_state.user_question = st.text_input("20글자 이상 입력해주세요.")

    if st.session_state.user_question == "초기화":
        st.session_state.conversation_history = []

    if len(st.session_state.user_question) > 20:
        user_input(st.session_state.user_question, api_key, st.session_state.conversation_history)

if __name__ == "__main__":
    main()
