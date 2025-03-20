import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough


# Load environment variables
load_dotenv()

# Initialize the language model
llm = OpenAI(
    temperature=0,  # Set to 0 for deterministic responses
    max_tokens=2000,  # Increase max_tokens to get full responses
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Template for generating a story title
title_template = PromptTemplate.from_template(
    "Write a captivating title for a story about {topic}."
)

# Template for writing the story based on the title
story_template = PromptTemplate.from_template(
    "You are a storyteller. Craft an engaging story with the title: '{title}'."
)

# Create chains for title and story generation
title_chain = title_template | llm

# Function to generate a story using the title
def generate_story(inputs):
    title = inputs["title"]
    story_prompt = story_template.format(title=title)
    return llm.invoke(story_prompt)

# Define the full chain
chain = (
    # Start with the input dictionary
    RunnablePassthrough.assign(
        # Generate title and add it to the input dictionary
        title=lambda x: title_chain.invoke({"topic": x["topic"]}).strip()
    )
    # Use the title to generate the story and add it to the dictionary
    .assign(
        story=lambda x: generate_story({"title": x["title"]})
    )
)

# Define the topic
topic = "a haunted lighthouse"

# Execute the chain
result = chain.invoke({"topic": topic})

# Output the results
print("Title:", result["title"])
print("Story:", result["story"])