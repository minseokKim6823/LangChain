import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from env_loader import load_env_file

load_env_file()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")

# llm = ChatOpenAI(
#     temperature=0.1,  # 창의성 (0.0 ~ 2.0)
#     max_tokens=2048,  # 최대 토큰수
#     model_name="gpt-4o",  # 모델명
# )
# prompt = PromptTemplate(
#     template = template, input_variables=["who"]
#     )
# 질의내용
# question = "세종대왕이 누구인지 설명해주세요"
# 질의
# result = llm.invoke(question)
# print(result.content)

# template = "{who}가 누구인지 설명해주세요"
#
# # 템플릿 완성
# prompt = PromptTemplate(
#         template=template, input_variables=['who']
#     )
# prompt = PromptTemplate(
#         template=template, input_variables=['history', 'input']
#     )

# llm 객체를 새로 선언하고, 프롬프트 템플릿과 llm 객체를 연결합니다.
# llm = ChatOpenAI(model_name="gpt-4o")
# # print(prompt.format(who="오바마"))
# llm_chain = prompt | llm
# result = llm_chain.invoke({"who":"이순신 장군"},
#                           config={"callbacks": [ConsoleCallbackHandler()]})
# print(result.content)


# 프롬프트 템플릿 생성

template = """아래는 사람과 AI의 친근한 대화입니다. AI의 이름은 챗봇입니다. 대화 문맥을 바탕으로 친절한 답변을 진행하세요.

Current Conversation:
{history}

Human: {input}
AI:"""

prompt = PromptTemplate(
        template=template, input_variables=['history', 'input']
    )

# llm 객체를 새로 선언하고, 프롬프트 템플릿과 llm 객체를 연결합니다.
llm = ChatOpenAI(model_name="gpt-4o")
chain = prompt | llm

store = {}
session_id = "test"

if session_id not in store:
    store[session_id] = ChatMessageHistory()
session_history = store[session_id]

with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: session_history,
    input_messages_key="input",
    history_messages_key="history"
)

result = with_message_history.invoke(
    {"input": "당신은 이름이 뭐예요?"},
    config={"configurable": {"session_id": "test"}},
)
print(result.content)

result = with_message_history.invoke(
    {"input": "푸른 바다를 주제로 감성적이고 짧은 시를 하나 지어주세요"},
    config={"configurable": {"session_id": "test"}},
)
print(result.content)

print (store)
