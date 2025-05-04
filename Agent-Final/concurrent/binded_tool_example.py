from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain_community.llms import Ollama
from langchain.agents.agent_types import AgentType

def is_credit_tool(input_str: str) -> str:
    return "12345"

credit_tool = Tool(
    name="Credut Score Checker",
    description="Takes the user name and gives the credit score of the user.",
    func=is_credit_tool,
)

model = "llama3:70b"
llm = Ollama(model=model)

tools = [credit_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

task_prompt = "Can you tell credit score of Abhinav? Use the tool to check."

response = agent.invoke(task_prompt)
print(response)
