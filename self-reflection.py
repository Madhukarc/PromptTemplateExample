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

prompt_template = """You are an AI assistant that uses self-reflection to refine a short story. Follow these steps:
1. Generate a short fictional story (3-5 sentences) based on the prompt {text}.
2. Reflect on whether the story is logically coherent, engaging, and clear.
3. Make necessary improvements based on your reflection.
4. Provide the final, refined version of the story.
Use the following format:
<thinking>
[Generate the short story.]
</thinking>
<reflection>
[Analyze whether the story makes sense. Identify inconsistencies, plot gaps, or lack of engagement.]
</reflection>
<adjustment>
[Revise the story to improve coherence and engagement.]
</adjustment>
<output>
[Provide the final, improved version of the story.]
</output>
Now, generate a short story using this process.
"""

# Create a PromptTemplate instance
prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

# Initialize the OpenAI LLM
llm = OpenAI(
    temperature=0,  # Set to 0 for deterministic responses
    max_tokens=3000,  # Increase max_tokens to get full response
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create a chain using the prompt and OpenAI model

chain = prompt | llm

# Provide input text
input_text = "Write a short story about a detective solving a mysterious case."

# Get the response
output = chain.invoke(input_text)

# Print the result
print("\n### Final Self-Consistent Solution ###\n", output)