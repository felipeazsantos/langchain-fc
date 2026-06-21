from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

load_dotenv()

long_text = """
    De tudo, ao meu amor serei atento
Antes, e com tal zelo, e sempre, e tanto
Que mesmo em face do maior encanto
Dele se encante mais meu pensamento.

Quero vivê-lo em cada vão momento
E em louvor hei de espalhar meu canto
E rir meu riso e derramar meu pranto
Ao seu pesar ou seu contentamento.

E assim, quando mais tarde me procure
Quem sabe a morte, angústia de quem vive
Quem sabe a solidão, fim de quem ama

Eu possa me dizer do amor (que tive):
Que não seja imortal, posto que é chama
Mas que seja infinito enquanto dure.
"""


splitter = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=70
)

parts = splitter.create_documents([long_text])

# for part in parts:
#     print(part.page_content)
#     print("-"*30)


llm = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

chain_summarize= load_summarize_chain(llm, chain_type="stuff")

result = chain_summarize.invoke({ "input_documents": parts})

print(result["output_text"])
