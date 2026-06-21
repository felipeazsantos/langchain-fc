from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

questionTemplate = PromptTemplate(
    input_variables=["name"],
    template="Hi, I`m {name} Tell me a joke with my name!"
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

chain = questionTemplate | model

result = chain.invoke({"name": "Felipe"})

print(result.content)