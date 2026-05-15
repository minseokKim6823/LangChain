import os
import urllib.request
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from env_loader import load_env_file

load_env_file()
if not os.getenv("OPENAI_API_KEY"):
  raise RuntimeError("OPENAI_API_KEY environment variable is required.")

urllib.request.urlretrieve("https://raw.githubusercontent.com/chatgpt-kr/openai-api-tutorial/main/ch06/test.txt", filename="test.txt")

with open("test.txt", encoding="utf-8") as f:
    file = f.read()
print('텍스트의 길이:', len(file))

text_splitter = SemanticChunker(OpenAIEmbeddings())
texts = text_splitter.create_documents([file])
print('분할된 청크의 수:', len(texts))

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3,
)
texts = text_splitter.create_documents([file])
print('분할된 청크의 수:', len(texts))

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="interquartile",
    breakpoint_threshold_amount=1.5
)
texts = text_splitter.create_documents([file])
print('분할된 청크의 수:', len(texts))
