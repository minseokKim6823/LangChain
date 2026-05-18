import os
import urllib.request
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
from env_loader import load_env_file

load_env_file()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")

loader = PyPDFLoader('2020_경제금융용어 700선_게시.pdf')
texts = loader.load_and_split()

texts = texts[13:]
texts = texts[:-1]

embedding = OpenAIEmbeddings(chunk_size=100)

vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding)
documents = vectordb._collection.get()['documents']
# print('청크의 개수 :', len(documents))
# print('-' * 50)
# print('0번 청크 출력 :', documents[0])

embeddings = vectordb._collection.get(include=['embeddings'])['embeddings']
# print('0번 청크의 임베딩 값 출력 :', embeddings[0])
# print('0번 청크의 임베딩 값의 길이 :', len(embeddings[0]))

metadatas = vectordb._collection.get()['metadatas']
# print('metadatas의 개수 :', len(metadatas))
# print('0번 청크의 출처 :', metadatas[0])

retriever = vectordb.as_retriever(search_kwargs={"k": 2}) #검색 시 유사한 문서 2개를 반환하도록 설정

docs = retriever.invoke("비트코인이 궁금해")
# print('유사 문서 개수 :', len(docs))
# print('--' * 20)
# print('첫번째 유사 문서 :', docs[0])
# print('두번째 유사 문서 :', docs[1])

template = """당신은 한국은행에서 만든 금융 용어를 설명해주는 금융쟁이입니다.
김민석 개발자가 만들었습니다. 주어진 검색 결과를 바탕으로 답변하세요.
검색 결과에 없는 내용이라면 답변할 수 없다고 하세요. 반말로 친근하게 답변하세요.
{context}

Question: {question}
Answer:
"""

def get_chatbot_response(input_text):
    chatbot_response = qa_chain(input_text)
    return chatbot_response['result']


prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type_kwargs={"prompt": prompt},
    retriever=retriever,
    return_source_documents=True)


# input_text = "비트코인에 대해서 궁금하당~"
# chatbot_response = qa_chain(input_text)
# result = get_chatbot_response(input_text)
# print(result)
# print(chatbot_response)

# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot(label = "경제 금융용어 챗봇")
#     msg = gr.Text


