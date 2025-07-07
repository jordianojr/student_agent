import ollama

# model that I used most for RAG
model = ollama.create(model='student_agent', from_='llama3.2:3b', system="You are a sophomore Computer Information Systems undergraduate.")