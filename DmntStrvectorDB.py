import os
import urllib.request
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from env_loader import load_env_file

load_env_file()
if not os.getenv("OPENAI_API_KEY"):
  raise RuntimeError("OPENAI_API_KEY environment variable is required.")


loader = PyPDFLoader('2023_북한인권보고서.pdf')
pages = loader.load_and_split()
print('청크의 수:', len(pages))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splitted_docs = text_splitter.split_documents(pages)
print('분할된 청크의 수:', len(splitted_docs))

# chunks = [splitted_doc.page_content for splitted_doc in splitted_docs]
# print('청크의 최대 길이 :',max(len(chunk) for chunk in chunks))
# print('청크의 최소 길이 :',min(len(chunk) for chunk in chunks))
# print('청크의 평균 길이 :',sum(map(len, chunks))/len(chunks))

db = Chroma.from_documents(splitted_docs, OpenAIEmbeddings(chunk_size=100))
print('문서의 수:', db._collection.count())

# question = '북한의 교육과정'
# docs =db.similarity_search(question)
# print('문서의 수:', len(docs))
# for doc in docs:
#   print(doc)
#   print('--' * 10)

faiss_db = FAISS.from_documents(splitted_docs, OpenAIEmbeddings())
print('문서의 수:', faiss_db.index.ntotal)

faiss_db.save_local('faiss_index')
new_db_faiss = FAISS.load_local('faiss_index',
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True)

question = '북한의 교육 과정'
docs = new_db_faiss.similarity_search(question)

for doc in docs:
  print(doc)
  print('--' * 10)