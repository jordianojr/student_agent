import json
import re
from typing import List, TypedDict
from langgraph.graph import StateGraph, END
import logging
import os
from dotenv import load_dotenv
import ollama
load_dotenv()

class SemanticChunker:
    def __init__(self):
        self.n_ctx = 10000


    def write_chunking_plan(self, content):
        '''
            Prompt to extract content ideas.

            Parameters:
            - content (String): Scraped and cleaned content

            Returns:
            - list: List of ideas found in the content
            '''
        # gemma-2-9b-it prompt
        summarize_template = '''
        <start_of_turn>user
        You are a diligent STEM student reviewing study notes for finals. Extract ALL important information as separate ideas.

        Extract from these slides:
        **FORMULAS & EQUATIONS:**
        - List every formula with variable definitions
        - Note when each formula applies (conditions/constraints)
        - Include units where relevant

        **KEY CONCEPTS & DEFINITIONS:**
        - Technical terms that will be tested
        - Theorems and their conditions
        - Physical/mathematical principles
        - Constants and their values

        **PROCEDURES & ALGORITHMS:**
        - Step-by-step methods
        - Decision trees for choosing approaches
        - Computational procedures

        **GENERAL IDEAS:**
        - Any other important points for exam prep

        <original_content>
        {content}
        </original_content>

        IMPORTANT: Format EVERY item as "Idea X:" followed by the BIG PICTURE idea. Number them sequentially.
        Break down complex ideas into multiple items if needed.
        Do not limit yourself to just 10 ideas; extract ALL relevant information.

        Example:
        Idea 1: Newton's second law: F = ma, where F is force (N), m is mass (kg), a is acceleration (m/s²)
        Idea 2: Kinematic equations apply only when acceleration is constant
        Idea 3: To solve projectile motion: separate x and y components, use appropriate kinematic equations for each

        <end_of_turn>
        <start_of_turn>model
        '''

        response = ollama.chat(
        model='qwen3:4b',
        messages=[
            {"role": "user", 
            "content": summarize_template.format(content=content)
            }
        ]
    )
        all_ideas = response['message']['content']
        print(f'Chunking plan created: \n{all_ideas}\n')

        pattern = r"Idea \d+: (.*)"
        matches = re.findall(pattern, all_ideas)
        list_of_ideas = [idea.strip() for idea in matches]

        return list_of_ideas

    def extract_chunk(self, idea, text):
        extract_template = '''
    <start_of_turn>user
    You are a thoughtful analyst tasked with reviewing a piece of writing and identifying sentences that directly support, explain, or relate to a specific idea.
    Your job is to extract exact sentences from the original content that are semantically related to the provided idea. These may reinforce the idea, give examples, expand on it, or express it in different words.
    Do not leave out any context that helps explain the sentences.
    Here is the original content:
    <original_content>
    {content}
    </original_content>
    And here is the target idea:
    <idea>
    {target_idea}
    </idea>

    Return the matching sentences in this JSON format STRICTLY:

    <format>
    ```json
    {{"related": [
    "...",
    "...",
    ...
    ]
    }}
    ```
    </format>

    Only include exact sentences from the original content. If no sentences match, return an empty list.
    <end_of_turn>
    <start_of_turn>model
    '''
        response = ollama.chat(
            model='qwen3:4b',
            messages=[
                {
                    "role": "user",
                    "content": extract_template.format(content=text, target_idea=idea)
                }
            ]
        )
        response = response['message']['content']
        # print(f'Extracted related sentences:\n{response}\n')
        chunk = ""
        try:
            # Look for JSON content between triple backticks
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If not found, try to extract the entire JSON object
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    print(f"No JSON found in response: {response}")
                    return ""
            
            parsed_feedback = json.loads(json_str)
            sentences = parsed_feedback['related']
            chunk = " ".join(sentences)
            return chunk
        except Exception as e:
            print(f"Error parsing feedback JSON: {e}")
            print(f"Response was: {response}")
            # Create a default feedback structure if parsing fails
            return ""

    
    def split_text(self, text):
        """
        Splits text into chunks using RecursiveCharacterTextSplitter.
        
        :param text: The input string to be split.
        :param chunk_size: Maximum size of each chunk.
        :param chunk_overlap: Number of overlapping characters between chunks.
        :return: List of text chunks.
        """
        chunks = []
        try:
            list_of_ideas = self.write_chunking_plan(text)
            print(f'List of ideas: {list_of_ideas}\n')
            # checked_ideas = self.check_idea_redundancy(list_of_ideas)
            # print(f'Checked ideas: {checked_ideas}\n')
            for idea in list_of_ideas:
                print(f'Extracting chunk for idea: {idea}')
                chunk = self.extract_chunk(idea, text)
                if chunk:
                    print(f'Extracted chunk: {chunk}\n')
                    chunks.append(idea + ". " + chunk)
        except Exception as e:
            print("********************************************** SEMANTIC CHUNKING FAILED **********************************************.")
            print(e)
        
        return chunks

if __name__ == "__main__":
    # Example
    chunker = SemanticChunker()
    text = """ Architectural Thinking I
CS 301: IT Solution Architecture
Week 1"""
    
    chunks = chunker.split_text(text)
    print("Semantic Chunks:")
    print("\n".join(chunks))
    # for i, chunk in enumerate(chunks, 1):
    #     print(f"{i}. {chunk}")
# Example usage of the SemanticChunker class