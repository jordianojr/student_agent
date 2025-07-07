import json
import re
import ollama
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import TypedDict, List, Dict
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage
from langgraph.graph import StateGraph, END
from agents.file_extractor import file_processor
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
    studied: List[str]
    get_knowledge: List[str]
    key_phrases: List[str]
    content: str
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
    "PLAN": """
            {question}

            You are an AI system that uses external documents to answer questions.

            Step 1: Read the above question carefully.
            Step 2: Decide whether answering this question requires additional information not found in your internal knowledge.
            Step 3: If external information is needed, generate a list of **concise and specific search queries** that would help retrieve the necessary information from a document database or knowledge base.

            Guidelines:
            - Each query should be short, factual, and optimized for retrieval.
            - Focus on key concepts, entities, or facts needed to answer the question.
            - Do not explain or justify your reasoning.
            - Do not include any additional output besides the JSON.

            Respond strictly in the following JSON format:

            ```json
            {{"get_knowledge": ["query 1", "query 2", "query 3"]}}
            ```
            """,    
            
    "RETRIEVE": """
            {knowledge_list}

            Read the above question list and pick out ONLY key phrases needed for knowledge retrieval.
            The key phrases will be used to perform knowledge retrieval on a vector database to retrieve relevant documents.
            The key phrases have to be related to answering the questions.

            Stop and think before responding.

            <example>
            Input:
            ["What is data analytics about?", "What are some data analytics libraries in Python?"]
            Response:
            {{"key_phrases": ["data analytics", "Python data analytics libraries"]}}
            <example>

            Do not include any other text or explanation in your response.
            Ensure no redundant phrases are included in the response.

            Provide your structured response in the JSON format below:  

            ```json
                {{"key_phrases": [List of key phrases to be used in knowledge retrieval]}}
            ```
            """,

    "ANSWER": """
            QUESTION: {question}

            CONTENT: {content}

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

    "CRITIQUE": """
            QUESTION: {question}

            CONTENT: {content}

            With the content given, read the above question and determine the difficulty level of answering the question.
            If the question is poor, critique how the question can be improved to be more aligned with the content.
            
            Stop and think before responding.

            Provide your structured response in the JSON format below:  

            ```json
                {{"comment": comment about the question alignment with the content}}
            ```
            """,
}

# Model reads the question and decides if it needs additional knowledge to answer
def plan_node(state: AgentState):
    """Plan the next steps based on the question."""
    response = ollama.chat(
        model='student_agent',
        messages=[
            {"role": "user", 
            "content": PROMPTS["PLAN"].format(question=state['question'])
            }
        ]
    )
    action = response['message']['content']
    try:
        # Look for JSON content between triple backticks
        json_match = re.search(r'```json\s*(.*?)\s*```', action, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If not found, try to extract the entire JSON object
            json_str = re.search(r'(\{.*\})', action, re.DOTALL).group(1)
                
        parsed_action = json.loads(json_str)
        print(f"PLAN: {parsed_action}")
        
        # Initialize get_knowledge with empty list by default
        state['get_knowledge'] = parsed_action.get('get_knowledge', [])
        print(f"get_knowledge: {state['get_knowledge']}")
        return state
    except Exception as e:
        print(f"Error parsing plan response: {e}")
        # Ensure get_knowledge exists in state even if parsing fails
        state['get_knowledge'] = []
        return state

def knowledge_retrieval_node(state: AgentState):
    """Retrieve knowledge needed to answer the question."""
    response = ollama.chat(
        model='student_agent',
        messages=[
            {
                "role": "user",
                "content": PROMPTS["RETRIEVE"].format(knowledge_list=json.dumps(state.get('get_knowledge', [])))
            }
        ]
    )

    key_phrases = response['message']['content']

    try:
        # Try to extract JSON from AI response
        json_match = re.search(r'```json\s*(.*?)\s*```', key_phrases, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = re.search(r'(\{.*\})', key_phrases, re.DOTALL).group(1)

        parsed_response = json.loads(json_str)
        key_phrases = parsed_response['key_phrases']
        state['key_phrases'] = key_phrases
        # Run the async tool call
        content = []
        for phrase in key_phrases:
            print(f"Calling RAG on key phrase: {phrase}")
            print(f"Student only studied for {state['studied']}")
            rag_content = file_processor.vector_search(phrase, state['studied'])
            print(f"Retrieved content: {rag_content}")
            content.append(rag_content)
        # state['content'] = asyncio.run(call_tools(key_phrases))
        state['content'] = content
        return state
    except Exception as e:
        print(f"Error in knowledge_retrieval_node: {e}")
        state['content'] = "Failed to retrieve knowledge."
        return state

def answer_node(state):
    """Generate an answer to the question based on retrieved content."""
    # Using retrieved content to answer the question
    response = ollama.chat(
        model='student_agent',
        messages=[{
            "role": "user",
            "content": PROMPTS["ANSWER"].format(question=state['question'], content=state['content'])
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
        model='student_agent',
        messages=[{
            "role": "user", 
            "content": PROMPTS["CRITIQUE"].format(question=state['question'], content=state['content'])
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

def rag_needed(state: AgentState) -> str:
    """Check if RAG is needed based on the state."""
    if len(state.get("get_knowledge", [])) > 0:
        print("RAG needed")
        return "retriever"
    print("RAG not needed")
    return "answer"

def generate_graph():
    """Generate the state graph for the agent."""
    builder = StateGraph(AgentState)
    builder.add_node("planner", plan_node)
    builder.add_node("retriever", knowledge_retrieval_node)
    builder.add_node("answer", answer_node)
    builder.add_node("critique", critique_node)

    builder.set_entry_point("planner")
    builder.add_conditional_edges(
        "planner", 
        rag_needed,
        {"answer": "answer", "retriever": "retriever"})
    builder.add_edge("retriever", "answer")
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
        'studied': agent_db['studied'],
        'get_knowledge': [],
        'content': "",
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
        'qn_id': question['qn_id'],
        'question': question['qn_options'],
        'final_answer': result['final_answer'][0],
        'justification': result['justification'],
        'confidence_score': result['confidence_score'],
        'comment': result['comment'],
        'is_correct': is_correct
    }
    fieldnames = ['student_name', 'studied', 'qn_id', 'question', 'final_answer', 'justification', 'confidence_score', 'comment', 'is_correct']
    csv_writer.write_to_csv("./results/" + csv_name, csv_data, fieldnames)

    return {
        'student_answer': result['final_answer'],
        'confidence_score': result['confidence_score'],
        'justification': result['justification'],
        'is_correct': is_correct,
    }

# for testing
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