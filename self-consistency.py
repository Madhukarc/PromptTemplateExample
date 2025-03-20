import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Create a prompt template
# prompt_template = """You are a math expert. Solve the quadratic equation: x² - 5x + 6 = 0 using three different methods.
# Follow a self-consistency approach by checking if the results match across methods.

# ### Method 1: Quadratic Formula
# Step 1: Recall the quadratic formula: x = (-b ± √(b² - 4ac)) / (2a)
# Step 2: Identify a, b, and c values.
# Step 3: Compute the discriminant and solve for x.

# ### Method 2: Factorization
# Step 1: Rewrite the quadratic equation in factored form.
# Step 2: Solve for x by setting each factor equal to zero.

# ### Method 3: Numerical Approximation
# Step 1: Use a numerical method (e.g., trial values, Newton-Raphson, or a solver).
# Step 2: Compare the computed values with previous solutions.

# ### Consistency Check:
# - Compare the answers obtained from all three methods.
# - If they match, confirm the result.
# - If they differ, analyze and resolve discrepancies.

# Now, execute each step and return a structured output.
# """

prompt_template = """Solve the quadratic equation: x² - 5x + 6 = 0
Generate multiple solution approaches:

1. Solve using the quadratic formula.
2. Solve using factorization.
3. Solve using numerical methods.
Show the step-by-step process for each method and compare the results for consistency.
Select the answer that appears most consistently across the methods.
"""

# Create a PromptTemplate instance
prompt = PromptTemplate(input_variables=[], template=prompt_template)

# Initialize the OpenAI LLM
llm = OpenAI(
    temperature=0,  # Set to 0 for deterministic responses
    max_tokens=2000,  # Increase max_tokens to get full response
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create a chain using the prompt and OpenAI model

chain = prompt | llm



# Get the response
output = chain.invoke({})

# Print the result
print("\n### Final Self-Consistent Solution ###\n", output)