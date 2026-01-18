from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

# 1️⃣ Load Hugging Face token
load_dotenv()

# 2️⃣ Free Hugging Face model (NO Nebius issue)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=300,
    temperature=0.3,
)

# 3️⃣ Chat wrapper
model = ChatHuggingFace(llm=llm)

# 4️⃣ Define output schema
schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
]

# 5️⃣ Structured parser
parser = StructuredOutputParser.from_response_schemas(schema)

# 6️⃣ Prompt template
template = PromptTemplate(
    template="Give me 3 facts about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# 7️⃣ Correct chain order
chain = template | model | parser

# 8️⃣ Run chain
result = chain.invoke({"topic": "the moon"})

# 9️⃣ Output
print(result)
