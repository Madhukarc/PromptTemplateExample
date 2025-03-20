import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Create a prompt template
prompt_template = "Classify the text into positive, negative, or neutral.: {text}"

# Create a PromptTemplate instance
prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

# Initialize the OpenAI LLM
llm = OpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create a chain using the prompt and OpenAI model

chain = prompt | llm

# Provide input text
input_text = "I would definitely like to visit your place."

# Get the response
output = chain.invoke(input_text)

# Print the result
print(output)