import os
from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain import hub

# --- 1. 设置您的 API 密钥和模型 ---
os.environ["NVIDIA_API_KEY"] = "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" 

llm = ChatOpenAI(
    model="meta/llama3-8b-instruct",
    openai_api_base="https://integrate.api.nvidia.com/v1",
    openai_api_key=os.environ.get("NVIDIA_API_KEY"),
    temperature=0.0,
)

# --- 2. 定义工具 ---
api_wrapper = WikipediaAPIWrapper(top_k_results=2)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tools = [wikipedia_tool]

# --- 3. 创建智能体 ---
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 4. 运行智能体 ---
print("AI 智能体已准备就绪 (LangChain版)，请输入您的问题 (输入 'exit' 或 'quit' 退出):")
while True:
    try:
        user_input = input("> ")
        if user_input.lower() in ["exit", "quit"]:
            print("再见！")
            break

        result = agent_executor.invoke({"input": user_input})

        print("\n----- Agent 回答 -----")
        print(result.get("output"))
        print("---------------------\n")

    except Exception as e:
        print(f"发生错误: {e}")
        break