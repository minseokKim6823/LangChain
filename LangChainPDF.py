import urllib.request
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader('2023_북한인권보고서.pdf')
pages = loader.load_and_split()
print('청크의 수:', len(pages))
print(pages[3])