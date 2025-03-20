import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# User inputs for credit score and income
#credit_score = 720  # Example credit score
#income = 70000  # Example income in dollars

# Take user input for credit score and income
credit_score = input("Enter Credit Score: ").strip()
income = input("Enter Income (in dollars): ").strip()


# Create a prompt template with placeholders
prompt_template = """Classify loan applications based on credit score and income.

Examples:
Credit Score = 750, Income = $80,000  // Approved
Credit Score = 680, Income = $50,000  // Conditionally Approved
Credit Score = 600, Income = $30,000  // Rejected

Now, classify the following application:
Credit Score = {credit_score}, Income = ${income}

Loan Decision:"""

# Create a PromptTemplate instance
prompt = PromptTemplate(input_variables=["credit_score", "income"], template=prompt_template)

# Initialize the OpenAI LLM
llm = OpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create a chain using the prompt and OpenAI model
chain = prompt | llm

# Invoke the chain with input variables
output = chain.invoke({"credit_score": credit_score, "income": income})

# Print the result
print("Loan Decision:", output)
