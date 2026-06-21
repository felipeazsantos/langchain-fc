from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.runnables import chain
load_dotenv()


@chain
def square(input_dict:dict) -> dict:
    x = input_dict["x"]
    return {"square_result": x * x}

questionTemplate = PromptTemplate(
    input_variables=["name"],
    template="Hi, I`m {name} Tell me a joke with my name!"
)


questionTemplate2 = PromptTemplate(
    input_variables=["square_result"],
    template="tell me about the number {square_result}"
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

chain = questionTemplate | model
chain2 = square | questionTemplate2 | model

result = chain2.invoke({"x": 10})

print(result.content)