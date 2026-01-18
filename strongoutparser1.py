from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

#first create the prompt
template1=PromptTemplate(
    template="Write a deatlis report  about {topic}",
    input_variables=["topic"]
)

template2=PromptTemplate(
    template="Write a 5 line summary on the following text ./n{text}",
    input_variables=["text"]
)

prompt=StrOutputParser()

chain= template1 | model |template2 | model |prompt

result=chain.invoke({"topic":"black hole"})
print(result)



