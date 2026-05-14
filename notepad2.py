import urllib.request
from langchain_text_splitters import RecursiveCharacterTextSplitter

urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

with open("2016-10-20.txt", encoding="utf-8") as f:
    file = f.read()
# print('텍스트의 길이:', len(file))

texts = text_splitter.create_documents([file])
# print('분할된 청크의 수:', len(texts))

print('첫 번째 청크의 내용:', texts[1])
