import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Create a prompt template
prompt_template = """You are an expert in environmental science. Use a Chain-of-Thought (CoT) reasoning approach to generate a well-structured paragraph explaining the impact of climate change on agriculture.

Think through the problem step by step using the following structured format:

<thinking>
Step 1: Introduce climate change and its relevance to agriculture.
Step 2: Explain key impacts (e.g., temperature rise, rainfall change, crop yield reduction).
Step 3: Provide an example of how farmers are adapting to these changes.
Step 4: Conclude with a forward-looking statement on sustainable agriculture.
</thinking>

<reasoning>
Analyze each step logically. If needed, refine the explanation for clarity.
</reasoning>

<output>
Generate the final well-structured paragraph based on your logical breakdown.
</output>

Now, generate the response:
"""

# Create a PromptTemplate instance
prompt = PromptTemplate(input_variables=[], template=prompt_template)

# Initialize the OpenAI LLM
llm = OpenAI(
    temperature=0,  # Set to 0 for deterministic responses
    max_tokens=3000,  # Increase max_tokens to get full response
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create a chain using the prompt and OpenAI model

chain = prompt | llm



# Get the response
output = chain.invoke({})

# Print the result
print(output)