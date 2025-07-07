from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from bson import ObjectId
from agents.student_rag import begin_answer
from agents.student_systemprompt import begin_answer as begin_answer_large
from agents.file_extractor import file_processor
from datetime import datetime
import uvicorn
import os
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch

app = FastAPI()
db = None
vector_store = None

# Add startup state tracking
app.state.is_ready = False

class CreateStudent(BaseModel):
    name: str
    studied: List[str] = []
    
class Student(BaseModel):
    id: str = Field(alias="_id")
    name: str
    studied: List[str] = []
    messages: List[Dict] = []

class Question(BaseModel):
    qn_id: str
    qn_week: int
    qn_stem: str
    qn_options: str
    correct_option: str

class QuestionRequest(BaseModel):
    qn_ids: List[str]

class Message(BaseModel):
    message: str
    
API_URL = "http://localhost:8000" # Default to localhost if not set

async def init_db():
    global db, vector_store
    try:
        # MongoDB Atlas connection string
        mongodb_url = os.getenv("MONGODB_URL")
        # Create a new client and connect to the server with ServerApi=1
        client = AsyncIOMotorClient(mongodb_url, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
        db = client.agents_db
        
        # Initialize vector store
        embeddings = OpenAIEmbeddings()
        collection_name = "documents"
        index_name = "vector_index"  # This should match your Atlas Search index name
        vector_store = MongoDBAtlasVectorSearch(
            collection=db[collection_name],
            embedding=embeddings,
            index_name=index_name
        )
        
        await client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas!")
        return True
    except Exception as e:
        print(f"Error connecting to MongoDB Atlas: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    db_success = await init_db()
    if db_success:
        app.state.is_ready = True
    else:
        # Still mark as ready if DB fails, as it might be optional for some endpoints
        app.state.is_ready = True
        print("Warning: Application starting without database connection")

@app.get("/_health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    if not app.state.is_ready:
        raise HTTPException(status_code=503, detail="Application is not ready")
    return {"status": "healthy"}

@app.post(
    "/create_student",
    status_code=201,
    responses={
        201: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {"additionalProp1": "string"}
                }
            }
        }
    }
)
async def create_agent(agent_post: CreateStudent):
    try:
        # Parse the agent_post string into a dict
        agent_data = agent_post
        print(f"Received agent data: {agent_data}")  # Debug log
        
        # Create new agent document
        agent = {
            "name": agent_data.name,
            "studied": agent_data.studied,
            "messages": []
        }
        print(f"Created agent document: {agent}")  # Debug log
        
        # Insert into MongoDB
        try:
            result = await db.agents.insert_one(agent)
            agent_id = str(result.inserted_id)
            print(f"Successfully inserted agent with ID: {agent_id}")  # Debug log
        except Exception as e:
            print(f"MongoDB insertion error: {e}")  # Debug log
            raise
        
        return {"student_id": agent_id}
    except Exception as e:
        print(f"Error in create_agent: {e}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_id}", response_model=Student)
async def get_agent(agent_id: str):
    try:
        agent = await db.agents.find_one({"_id": ObjectId(agent_id)})
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        # Convert ObjectId to string for JSON serialization
        agent["_id"] = str(agent["_id"])
        return agent
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/all_agents", response_model=List[Student])
async def get_all_agents():
    try:
        agents = []
        async for agent in db.agents.find():
            agent["_id"] = str(agent["_id"])  # Convert ObjectId to string
            agents.append(agent)
        return agents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/all_questions", response_model=List[Question])
async def get_all_questions():
    try:
        questions = []

        # toggle between questions and questions_manual for LLM questions/manual questions
        async for question in db.questions_manual.find():
        # async for question in db.questions.find():

            questions.append(question)
        questions.sort(key=lambda x: x["qn_id"])  # Sort questions by qn_id
        return questions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/all_questions_id", response_model=List[str])
async def get_all_questions_id():
    try:
        questions = []

        # toggle between questions and questions_manual for LLM questions/manual questions
        async for question in db.questions_manual.find({}, {"qn_id": 1}):
        # async for question in db.questions.find({}, {"qn_id": 1}):
        
            questions.append(str(question["qn_id"]))
        questions.sort()  # Sort the question IDs
        return questions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/agents/{agent_id}", status_code=204)
async def delete_agent(agent_id: str):
    try:
        result = await db.agents.delete_one({"_id": ObjectId(agent_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Agent not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/answer/{agent_id}", status_code=201)
async def send_message(agent_id: str, message: Message):
    try:
        # Check if agent exists
        agent = await db.agents.find_one({"_id": ObjectId(agent_id)})
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Run the research process and get the final draft
        final_draft = begin_answer(question=message.message, agent_db=agent, csv_name="send_query_function.csv")

        # Store the result in MongoDB
        await db.agents.update_one(
            {"_id": ObjectId(agent_id)},
            {"$push": {"messages": {
                "query": message.message,
                "response": final_draft,
                "timestamp": datetime.utcnow()
            }}}
        )

        # Return the draft in the expected format
        return {
            "response": final_draft
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/answer_questions/{agent_id}", status_code=201)
async def answer_questions(agent_id: str, request: QuestionRequest):
    try:
        # Check if the application is ready
        if not app.state.is_ready:
            raise HTTPException(status_code=503, detail="Application is not ready")
        
        # Get qn_ids from the request body
        qn_ids = request.qn_ids
        
        # Validate questions input
        if not qn_ids or not isinstance(qn_ids, list):
            raise HTTPException(status_code=400, detail="Invalid questions input")
        
        agent_db = await db.agents.find_one({"_id": ObjectId(agent_id)})
        if not agent_db:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent_name = agent_db.get("name", "Unknown Agent")
        csv_name = f"{agent_name}_attempt.csv"
        
        # Run the research process for each question
        results = []
        for qn_id in qn_ids:
            # Fetch the question from the database
            logging.info(f"Fetching question with ID: {qn_id}")
            
            # toggle between questions and questions_manual for LLM questions/manual questions
            question = await db.questions_manual.find_one({"qn_id": qn_id})
            # question = await db.questions.find_one({"qn_id": qn_id})
            
            if not question:
                raise HTTPException(status_code=404, detail=f"Question with ID {qn_id} not found")
            
            logging.info(f"Processing question: {question['qn_id']}")
            final_draft = begin_answer(question=question, agent_db=agent_db, csv_name=csv_name)
            results.append({
                "qn_id": qn_id,
                "question": question['qn_options'],
                "response": final_draft
            })
            logging.info(f"Processed question {qn_id}")
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/files", status_code=200)
async def update_agent_files(files: List[UploadFile] = File(...)):
    try:
        # Log the incoming request
        logging.info(f"Number of files: {len(files)}")
        
        # Initialize file processor's database connection
        mongodb_url = os.getenv("MONGODB_URL")
        file_processor.init_db(mongodb_url)
        
        # Process and store files
        logging.info("Starting file processing")
        await file_processor.process_files(files)
        logging.info("File processing completed successfully")
        
        return {"message": "Files processed successfully"}
    except FileNotFoundError as e:
        logging.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logging.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error processing files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)