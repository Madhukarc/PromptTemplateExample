import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize the language model
llm = OpenAI(
    temperature=0,  # Set to 0 for deterministic responses
    max_tokens=2000,  # Increase max_tokens to get full responses
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create the ReAct prompt template
react_template = PromptTemplate.from_template("""
You are an AI assistant that solves logic puzzles using a ReAct approach. Follow these steps:
 **Think through the problem logically** step by step.  
 **Simulate an action** (e.g., filling, emptying, or transferring water between jars).  
 **Observe the result** of the action.  
 **Adjust** the reasoning if needed.  
 **Continue iterating** until you find a valid solution.  
 **Provide the final solution.**
Use the following structured format:
<thinking>
Step 1: Identify the constraints.  
Step 2: Consider different ways to measure 4 liters using the jars.  
</thinking>
<action>
[Perform an action, such as filling or transferring water.]
</action>
<observation>
[Observe what changed after the action.]
</observation>
<adjustment>
[Modify the approach if necessary.]
</adjustment>
<output>
[Provide the final correct solution.]
</output>
Now, solve the following puzzle:  
"{problem}"
""")

# Define the puzzle
puzzle = "You have a 5-liter jar and a 3-liter jar. How can you measure exactly 4 liters of water?"

# Format the prompt with our problem
formatted_prompt = react_template.format(problem=puzzle)

# Get the response from the LLM
response = llm.invoke(formatted_prompt)

# Print the result
print(response)


