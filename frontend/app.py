import gradio as gr
import requests
import json
from typing import List
import os
import concurrent.futures

API_URL = "http://localhost:8000"

# Define common study subjects
STUDY_OPTIONS = [
    "Week1.pptx",
    "Week2.pptx",
    "Week3.pptx",
    "Week4.pptx",
    "Week5.pptx",
    "Week6.pptx",
    "Week9.pptx",
    "Week10.pptx",
    "Week11.pptx",
    ]

class StudentAgentUI:
    def __init__(self):
        self.current_agent_id = None
        self.question_ids = {}
        self.agent_ids = {}  # Initialize as dictionary instead of list
        self.load_data()

    def load_data(self):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both API calls concurrently
                questions_future = executor.submit(requests.get, f"{API_URL}/all_questions")
                agents_future = executor.submit(requests.get, f"{API_URL}/all_agents")
                
                # Get results from both futures
                questions_response = questions_future.result()
                agents_response = agents_future.result()
                
                # Process questions response
                if questions_response.status_code == 200:
                    questions = questions_response.json()
                    self.question_ids = {question['qn_id']: {'question': question['qn_options'], 'answer': question['correct_option']} for question in questions}
                
                # Process agents response
                if agents_response.status_code == 200:
                    agents = agents_response.json()
                    self.agent_ids = {agent['name']: agent['_id'] for agent in agents}

        except Exception as e:
            print(f"Error loading data: {str(e)}")

    def create_agent(self, name: str, studied: List[str]) -> str:
        try:
            # Prepare the agent data
            agent_data = {"name": name, "studied": studied}
            response = requests.post(
                f"{API_URL}/create_student",
                json=agent_data
            )
            if response.status_code == 201:
                self.current_agent_id = response.json().get("student_id")
                return f"Agent created successfully! ID: {self.current_agent_id}"
            else:
                return f"Error creating agent: {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_agent(self, agent_id: str) -> str:
        try:
            response = requests.get(f"{API_URL}/agents/{agent_id}")
            if response.status_code == 200:
                return json.dumps(response.json(), indent=2)
            else:
                return f"Error getting agent: {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

    def delete_agent(self, agent_id: str) -> str:
        try:
            response = requests.delete(f"{API_URL}/agents/{agent_id}")
            if response.status_code == 204:
                return "Agent deleted successfully!"
            else:
                return f"Error deleting agent: {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

    def send_query(self, agent_id: str, query: str) -> str:
        try:
            response = requests.post(
                f"{API_URL}/answer/{agent_id}",
                json={"message": query}
            )
            if response.status_code == 201:
                return response.json().get("response", "No response field in API reply.")
            else:
                return f"Error sending query: {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

    def submit_selections(self, selected_agent: str, selected_questions: List[str]):
        try:
            if not selected_agent or not selected_questions:
                return "Please select an agent and at least one question."
            
            agent_id = self.agent_ids.get(selected_agent)
            if not agent_id:
                return f"Agent '{selected_agent}' not found."
            
            # Send agent_id as path parameter and qn_ids as JSON body
            response = requests.post(
                f"{API_URL}/answer_questions/{agent_id}",
                json={"qn_ids": selected_questions},  # Changed from data= to json=
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 201:
                return response.json()
            else:
                return f"Error submitting selections: {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"
        
    def upload_files(self, files: List) -> str:
        try:
            if not files:
                return "No files selected."
            files_data = [("files", (os.path.basename(f.name), f, "application/octet-stream")) for f in files]
            response = requests.post(
                f"{API_URL}/files",
                files=files_data
            )
            if response.status_code == 200:
                return "Files uploaded and processed successfully!"
            else:
                return f"Error uploading files: {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_all_agents(self):
        try:
            response = requests.get(f"{API_URL}/all_agents")
            if response.status_code == 200:
                agents = response.json()
                self.agent_ids = {agent['name']: agent['_id'] for agent in agents}
                return self.create_agent_cards(agents)
            else:
                return gr.HTML("<div style='color: red;'>Error fetching agents</div>")
        except Exception as e:
            return gr.HTML(f"<div style='color: red;'>Error: {str(e)}</div>")

    def create_agent_cards(self, agents):
        if not agents:
            return gr.HTML("<div>No agents found</div>")
        
        cards_html = "<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; padding: 20px;'>"
        for agent in agents:
            agent_id = agent.get('_id', 'N/A')
            name = agent.get('name', 'Unnamed')
            studied = agent.get('studied', [])
            studied_text = '<br>'.join(studied) if studied else 'No subjects studied'
            
            card = f"""
            <div style='border: 1px solid black; background: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='margin-top: 0; color: #000000;'>{name}</h3>
                <p style='color: #000000; margin: 5px 0;'><strong>ID:</strong> {agent_id}</p>
                <div style='margin-top: 10px;'>
                    <p style='color: #000000; margin: 5px 0;'><strong>Studied:</strong></p>
                    <p style='color: #000000; margin: 5px 0;'>{studied_text}</p>
                </div>
            </div>
            """
            cards_html += card
        cards_html += "</div>"
        return gr.HTML(cards_html)

    def get_all_questions(self):
        try:
            response = requests.get(f"{API_URL}/all_questions")
            if response.status_code == 200:
                questions = response.json()
                return self.create_question_cards(questions)
            else:
                return gr.HTML("<div style='color: red;'>Error fetching questions</div>")
        except Exception as e:
            return gr.HTML(f"<div style='color: red;'>Error: {str(e)}</div>")

    def create_question_cards(self, questions):
        if not questions:
            return gr.HTML("<div>No questions found</div>")
        
        cards_html = "<div style='display: flex; flex-direction: column; gap: 20px; padding: 20px;'>"
        for question in questions:
            question_id = question.get('qn_id', 'N/A')
            question_text = question.get('qn_options', 'No question text')
            answer = question.get('correct_option', 'No answer available')
            question_week = question.get('qn_week', 'No week available')
            
            card = f"""
            <div style='border: 1px solid black; background: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%;'>
                <h3 style='margin-top: 0; color: #000000;'>Question</h3>
                <p style='color: #000000; margin: 5px 0;'><strong>ID:</strong> {question_id}</p>
                <div style='margin-top: 10px;'>
                    <p style='color: #000000; margin: 5px 0;'><strong>Week {question_week}</strong></p>
                    <p style='color: #000000; margin: 5px 0;'><strong>Question:</strong></p>
                    <p style='color: #000000; margin: 5px 0;'>{question_text}</p>
                    <p style='color: #000000; margin: 5px 0;'><strong>Correct answer:</strong></p>
                    <p style='color: #000000; margin: 5px 0;'>{answer}</p>
                </div>
            </div>
            """
            cards_html += card
        cards_html += "</div>"
        return gr.HTML(cards_html)

# Create instance of the UI handler
agent_ui = StudentAgentUI()

# Create the Gradio interface
with gr.Blocks(title="Student Agent Interface") as demo:
    gr.Markdown("# Student Agent Interface")
    
    with gr.Tab("All Agents"):
        all_agents_output = gr.HTML()
        all_agents_btn = gr.Button("Refresh Agents")
        all_agents_btn.click(
            fn=agent_ui.get_all_agents,
            outputs=all_agents_output
        )
        # Load agents automatically when page loads
        demo.load(
            fn=agent_ui.get_all_agents,
            outputs=all_agents_output
        )

    with gr.Tab("Create Agent"):
        with gr.Row():
            name_input = gr.Textbox(label="Agent Name")
            studied_input = gr.CheckboxGroup(
                label="Studied Weeks",
                choices=STUDY_OPTIONS,
                value=[],
                interactive=True
            )
        create_btn = gr.Button("Create Agent")
        create_output = gr.Textbox(label="Result")
        create_btn.click(
            fn=agent_ui.create_agent,
            inputs=[name_input, studied_input],
            outputs=create_output
        )

    with gr.Tab("Get Agent"):
        with gr.Row():
            get_agent_id = gr.Textbox(label="Agent ID")
            get_btn = gr.Button("Get Agent Details")
        get_output = gr.Textbox(label="Agent Details", lines=10)
        get_btn.click(
            fn=agent_ui.get_agent,
            inputs=get_agent_id,
            outputs=get_output
        )

    with gr.Tab("Send Query"):
        with gr.Row():
            query_agent_id = gr.Textbox(label="Agent ID")
            query_input = gr.Textbox(label="Query", lines=3)
        query_btn = gr.Button("Send Query")
        query_output = gr.Textbox(label="Response", lines=10)
        query_btn.click(
            fn=agent_ui.send_query,
            inputs=[query_agent_id, query_input],
            outputs=query_output
        )

    with gr.Tab("Upload Files"):
        with gr.Row():
            upload_files_input = gr.File(label="Upload Files", file_count="multiple")
        upload_files_btn = gr.Button("Upload Files")
        upload_files_output = gr.Textbox(label="Result")
        upload_files_btn.click(
            fn=agent_ui.upload_files,
            inputs=upload_files_input,
            outputs=upload_files_output
        )

    with gr.Tab("All Questions"):
        all_questions_output = gr.HTML()
        all_questions_btn = gr.Button("Refresh Questions")
        all_questions_btn.click(
            fn=agent_ui.get_all_questions,
            outputs=all_questions_output
        )
        demo.load(
            fn=agent_ui.get_all_questions,
            outputs=all_questions_output
        )
    with gr.Tab("Answer Questions"):
        agents_id_output = gr.Radio(
            label="Select Agent IDs",
            choices=list(agent_ui.agent_ids.keys()),
            value=[],
            interactive=True
        )
        question_ids_output = gr.CheckboxGroup(
            label="Select Question IDs",
            choices=list(agent_ui.question_ids.keys()),
            value=[],
            interactive=True
        )
        with gr.Row():
            select_all_btn = gr.Button("Select All Questions")
            answer_btn = gr.Button("Answer Selected Questions")

        answer_output = gr.Textbox(label="Answer Result", lines=10)

        def select_all_questions():
            return list(agent_ui.question_ids.keys())
            
        def submit_selections(selected_agent, selected_questions):
            print("Selected Agent:", selected_agent)
            print("Selected Questions:", selected_questions)
            response = agent_ui.submit_selections(selected_agent, selected_questions)
            return response
        
        select_all_btn.click(
            fn=select_all_questions,
            outputs=question_ids_output
        )
        answer_btn.click(
            fn=submit_selections,
            inputs=[agents_id_output, question_ids_output],
            outputs=answer_output
        )


demo.launch(share=True)