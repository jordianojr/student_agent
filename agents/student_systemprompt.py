import json
import re
import ollama
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import TypedDict, List, Dict
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from agents.function_tool import csv_writer
import os
import logging

_ = load_dotenv()

embeddings = OpenAIEmbeddings()
mongo_client = MongoClient(os.environ["MONGODB_URL"])
collection = mongo_client.agents_db.documents

class AgentDB(BaseModel):
    id: str = Field(alias="_id")
    name: str
    studied: List[str] = []
    messages: List[Dict] = []

class AgentState(TypedDict):
    """State definition for the agent's workflow."""
    question: str
    model: str
    final_answer: str
    justification: str
    confidence_score: float
    comment: str

class Question(BaseModel):
    qn_id: str
    qn_week: int
    qn_stem: str
    qn_options: str
    correct_option: str

PROMPTS = {
    "ANSWER": """
            QUESTION: {question}

            Read and answer the above multiple-choice question and use the content as textbook knowledge.
            There is only ONE correct answer.

            If you are not sure about the answer, please select the best possible answer even if you have to guess.
            The confidence score should reflect how confident you are about your answer.
            If you are not confident, please provide a low confidence score.
            If you are very confident, please provide a high confidence score.
            The answer should be in the format of a list with one element, which is the correct option letter (e.g., ["A"]).
            For justification, provide a brief explanation of how you arrived at the answer based on the content provided.
            Provide quotes from the content to support your answer if possible.

            Stop and think before responding.
            Provide your structured response in the JSON format below:
            ```json
            {{"final_answer": ["<correct option letter>"], "confidence_score": 0.0, "justification": "how you arrived at the answer with references to the content"}}
            ```
            """,

    "RECHECK_ANSWER": """ """,

    "CRITIQUE": """
            QUESTION: {question}

            Read the above question and determine the difficulty level of answering the question.
            If the question is poor, critique how the question can be improved to be more aligned with the content in your system prompt.
            
            Stop and think before responding.

            Provide your structured response in the JSON format below strictly:  

            ```json
                {{"comment": comment about the question alignment with the content}}
            ```
            """,
}

def answer_node(state):
    """Generate an answer to the question based on retrieved content."""
    # Using retrieved content to answer the question
    response = ollama.chat(
        model=state['model'],
        messages=[{
            "role": "user",
            "content": PROMPTS["ANSWER"].format(question=state['question'])
        }]
    )
    answer = response['message']['content']
    
    try:
        # Try to extract JSON from AI response
        json_match = re.search(r'```json\s*(.*?)\s*```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback: look for JSON-like structure
            json_match = re.search(r'(\{.*\})', answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                raise ValueError("No JSON found in response")
        
        # Clean the JSON string by removing control characters
        json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)
        
        # Parse the JSON
        parsed_answer = json.loads(json_str)
        
        # Extract values with defaults
        answer = parsed_answer.get('final_answer', ['Unknown'])
        confidence_score = parsed_answer.get('confidence_score', 0.0)
        justification = parsed_answer.get('justification', 'No justification provided')
        
        print(f"ANSWER: {answer}, CONFIDENCE: {confidence_score}")
        
        state['final_answer'] = answer
        state['confidence_score'] = confidence_score
        state['justification'] = justification
        
        return state
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error in answer_node: {e}")
        print(f"Raw response: {answer}")
        print(f"Extracted JSON string: {json_str if 'json_str' in locals() else 'None'}")
        state['final_answer'] = ["Error: Invalid JSON response"]
        state['confidence_score'] = 0.0
        state['justification'] = f"JSON parsing failed: {str(e)}"
        return state
        
    except Exception as e:
        print(f"General error in answer_node: {e}")
        print(f"Raw response: {answer}")
        state['final_answer'] = ["Error in answer_node"]
        state['confidence_score'] = 0.0
        state['justification'] = f"Processing failed: {str(e)}"
        return state
    
def critique_node(state: AgentState):
    """Critique the answer for accuracy and completeness."""
    response = ollama.chat(
        model=state['model'],
        messages=[{
            "role": "user", 
            "content": PROMPTS["CRITIQUE"].format(question=state['question'])
            }]
    )
    critique = response['message']['content']
    try:
        # Try to extract JSON from AI response
        json_match = re.search(r'```json\s*(.*?)\s*```', critique, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = re.search(r'(\{.*\})', critique, re.DOTALL).group(1)
        
        # Clean the JSON string by removing control characters and properly escaping
        json_str = json_str.encode('utf-8').decode('unicode_escape')
        json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)
        
        parsed_critique = json.loads(json_str)
        critique = parsed_critique['comment']
        print(f"COMMENT: {critique}")
        state['comment'] = critique
        return state
    except Exception as e:
        print(f"Error in critique_node: {e}")
        state['comment'] = "Failed to comment."
        return state

def generate_graph():
    """Generate the state graph for the agent."""
    builder = StateGraph(AgentState)
    builder.add_node("answer", answer_node)
    builder.add_node("critique", critique_node)

    builder.set_entry_point("answer")

    builder.add_edge("answer", "critique")
    builder.add_edge("critique", END)

    return builder.compile()

def begin_answer(question: Question, agent_db: AgentDB, csv_name: str):
    """Start the research process with the given question."""
    graph = generate_graph()
    thread = {"configurable": {"thread_id": "1"}}
    final_state = None
    initial_state = {
        'question': question['qn_options'],
        'model': "weak_student",
        'final_answer': "",
        'justification': "",
        'confidence_score': 0.0,
        'comment': ""
    }
    
    for state in graph.stream(initial_state, thread):
        print(f"Current state: {state}")
        final_state = state
    
    result = final_state.get('critique')
    logging.info("LLM answer:" + result['final_answer'][0])
    logging.info("Correct answer:" + question['correct_option'])

    is_correct = False
    if len(final_state['critique']['final_answer']) == 1 and final_state['critique']['final_answer'][0] == question['correct_option']:
        is_correct = True

    csv_data = {
        'student_name': agent_db['name'],
        'studied': agent_db['studied'],
        'model': "weak_student",
        'qn_id': question['qn_id'],
        'question': question['qn_options'],
        'final_answer': result['final_answer'][0],
        'justification': result['justification'],
        'confidence_score': result['confidence_score'],
        'comment': result['comment'],
        'is_correct': is_correct
    }
    fieldnames = ['student_name', 'studied', 'model', 'qn_id', 'question', 'final_answer', 'justification', 'confidence_score', 'comment', 'is_correct']
    # csv_writer.write_to_csv("./test/" + csv_name, csv_data, fieldnames)
    csv_writer.write_to_csv("./results/" + "testing_result.csv", csv_data, fieldnames)


    return {
        'student_answer': result['final_answer'],
        'confidence_score': result['confidence_score'],
        'justification': result['justification'],
        'is_correct': is_correct,
    }

# if __name__ == "__main__":
#     chat_graph = generate_graph()
#     print(chat_graph.get_graph().draw_ascii())
#     result = begin_answer(Question(
#         qn_id="123",
#         qn_week=2,
#         qn_stem="What is the primary purpose of data analytics?",
#         qn_options="A. To collect data B. To analyze data C. To visualize data D. To store data E. To delete data",
#         correct_option="B"
#     ), AgentDB(
#         _id="123",
#         name="StudentA",
#         studied=["Week2.pptx"],
#     )
#     , csv_name="student_test.csv"
#     )
#     print("\nFINAL RESULT:")
#     print(f"Answer: {result.get('student_answer')}")
#     print(f"Justification: {result.get('justification')}")

    