from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# 1️⃣ Load .env
load_dotenv()

# 2️⃣ Hugging Face model (FREE, no Nebius issue)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=300,
    temperature=0.3
)

# 3️⃣ Chat wrapper
model = ChatHuggingFace(llm=llm)

# 4️⃣ JSON parser
parser = JsonOutputParser()

# 5️⃣ Prompt
prompt = PromptTemplate(
    template="Give me 5 facts about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# 6️⃣ Correct chain order
chain = prompt | model | parser

# 7️⃣ Invoke
result = chain.invoke({"topic": "the moon"})

# 8️⃣ Print
print(result)
