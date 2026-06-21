from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression and returns the result"""
    try:
        result = eval(expression)
    except Exception as e:
        return f"Error: {e}"
    return str(result)


@tool("web_search_mock")
def web_search_mock(query: str) -> str:
    """Mock web search function. Returns a hardcoded result"""
    data = {
        "Brasil": "Brasília",
        "Argentina": "Buenos Aires",
        "Chile": "Santiago",
        "Uruguai": "Montevidéu",
        "Paraguai": "Assunção",
        "Bolívia": "Sucre",
        "Peru": "Lima",
        "Colômbia": "Bogotá",
        "Venezuela": "Caracas",
        "Equador": "Quito",
        "México": "Cidade do México",
        "Estados Unidos": "Washington, D.C.",
        "Canadá": "Ottawa",
        "Portugal": "Lisboa",
        "Espanha": "Madri",
        "França": "Paris",
        "Itália": "Roma",
        "Alemanha": "Berlim",
        "Reino Unido": "Londres",
        "Japão": "Tóquio"
    }

    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"The capital of {country} is {capital}"

    return "I don`t know the capital of that country"


llm = ChatOpenAI(model="gpt-5-mini", disable_streaming=True)
tools = [calculator, web_search_mock]

prompt = PromptTemplate.from_template(
    """
    Answer the following questions as best as you can. You have access to the following tools.
    Only use the information you get from the tools, even if you know the answer.
    If the information is not provided by the tools, say you don`t know.
    
    {tools}
    
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Rules:
    - If you choose an Action, do NOT include Final Answer in the same step.
    - After Action and Action Input, stop and wait for Observation.
    - Never search the internet. Only use the tools provided.
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """
)

agent_chain = create_react_agent(llm, tools, prompt, stop_sequence=False)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_chain,
    tools=tools,
    verbose=True,
    handle_parsing_errors="Invalid format. Either provide an Action with Action Input, or a Final Answer only.",
    max_iterations=5,
)

# print(agent_executor.invoke({ "input": "Whats is the capital of Equador"}))
print(agent_executor.invoke({ "input": "How much is 10 + 10"}))
