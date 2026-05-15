import urllib.request
from langchain_text_splitters import RecursiveCharacterTextSplitter

urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")

# RecursiveCharacterTextSplitter 길이에 맞추어 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
with open("2016-10-20.txt", encoding="utf-8") as f:
    file = f.read()
# print('텍스트의 길이:', len(file))

texts = text_splitter.create_documents([file])
# print('분할된 청크의 수:', len(texts))

#Document(page_content="텍스트")의 형태
print(texts[1])
#원문 출력
print(texts[1].page_content)

print('1번 청크의 길이:', len(texts[1].page_content))
print('2번 청크의 길이:', len(texts[2].page_content))

