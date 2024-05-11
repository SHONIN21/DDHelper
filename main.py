import streamlit as st
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
import os
# from dotenv import load_dotenv
import tempfile
import shutil

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# load_dotenv()  # 環境変数を読み込む
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# 環境変数からAPIキーを読み込む
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


st.set_page_config(
    page_title="DD Document Review Helper",
    page_icon="./DALL·E 2024-04-08 10.33.03 - A sleek and modern logo design for an application called 'DD Document Review Helper'. The logo should feature two overlapping 'D' letters, stylized to.jpg",
    layout="wide"
)

col1, col2 = st.columns(2)
with col1:
    st.title('DD Document Review Helper')
    st.markdown("""
    ====================
    ### Who's problem?
    #### → Law firms and junior lawyers
    ====================
    ### When does it happen?
    #### → When conducting due diligence projects
    ====================                       
    ### What’s the problem?
    #### → Massive document review being troublesome
    ====================                      
    ### How to solve?
    #### → By simply uploading documents and questions on the applications, it provides answers and the basis locations (however, human check is required).This speed up the process of document reviewing
    ====================            
    """)

with col2:
    # Excelファイルをアップロードするためのウィジェット
    uploaded_file = st.file_uploader("Please upload an excel file of the checklist items.", type=["xlsx"])

    # ファイルがアップロードされた場合
    if uploaded_file is not None:
        # Pandasを使用してExcelファイルを読み込む
        rules_df = pd.read_excel(uploaded_file)
        st.write(rules_df)

        uploaded_state_file = st.file_uploader("Upload a regulatory document for the company being surveyed.", type=["pdf"])
        if uploaded_state_file is not None:
            # 一時ファイルを作成
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                # アップロードされたファイルの内容を一時ファイルに書き込む
                shutil.copyfileobj(uploaded_state_file, tmp)
                tmp_path = tmp.name  # 一時ファイルのパスを保存

            # 一時ファイルのパスを PyPDFLoader に渡す
            loader = PyPDFLoader(tmp_path)
            pages = loader.load_and_split()
            
            rules_df["Answer"] = None
            rules_df["BasisPageNumber"] = None
            rules_df["BasisContents"] = None
            
            # LangChain と OpenAI の設定
            llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-3.5-turbo")
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            vectorstore = Chroma.from_documents(pages, embedding=embeddings)     
            pdf_qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
            for index, row in rules_df.iterrows():
                query1 = f"You are a junior lowyer and your boss is going to give you some review task based on [document].Read through this document throughly, then answer the following question based on what you read. Question: {rules_df['Question'][index]}"
                chat_history = []
                result1 = pdf_qa({"question": query1, "chat_history": chat_history})
                # 回答をDataFrameに保存
                rules_df["Answer"][index] = result1["answer"]
                rules_df["BasisPageNumber"][index] = [doc.metadata['page'] for doc in result1['source_documents']][:2]
                rules_df["BasisContents"][index] = [doc.page_content for doc in result1['source_documents']][:2]
                # rules_df["BasisContents"][index] = str(result1['source_documents'])
            st.write(rules_df)
            vectorstore.delete_collection()
            vectorstore.persist()
            
            # 一時ファイルを削除
            os.unlink(tmp_path)
