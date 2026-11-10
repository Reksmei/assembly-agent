from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from uuid import uuid4
from tool import retrieve_relevant_info
import base64
import os
import pathlib
import shutil
import uvicorn
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from document_embeddings import main
from transformers import pipeline
import io
import numpy as np
import scipy.io.wavfile

app = FastAPI()

# Run the script to embedding the instruction manual
main()

# Initialize the model that powers the agent
model = ChatOllama(
    model="gemma4:e2b",
    num_predict=512,
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)

system_prompt = '''
You are a helpful assistant and handyman, who helps a user assemble a complex cabinet from Ikea.
When the user asks a question, use their text input and uploaded image, if provided to advise them
on what the next steps are, or what they may have missed and need to redo, or what parts they need/are missing.
Do not use over complex language, to make it as simple as possible for the user
Use the retrieve_relevant_info tool to help you by interpreting the user's query and then write your query to get relevant info from the instruction manual.
'''

# Wrap the model and tool into an agent and create an in-memory session
agent = create_agent(
    model=model,
    tools=[retrieve_relevant_info],
    checkpointer=InMemorySaver(),
    system_prompt=system_prompt
)

# Load a lightweight tts generator from Hugging Face
tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")

@app.post("/start_session")
def start_session():
    '''
    Starts a session and returns a session id to frontend to send in future interactions,
    so that the agent stores conversation in memory
    '''
    session_id = f"session-{uuid4()}"
    os.makedirs(f"uploads-{session_id}", exist_ok=True)

    return {
        "status": "Session Started",
        "session_id": session_id
    }

@app.post("/upload_image")
async def upload(session_id: str, request: Request, file: UploadFile = File(...)):
    
    # Look up the extension for the declared content type; None if not allowed
    ALLOWED = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    }

    ext = ALLOWED.get(file.content_type)
    if not ext:
        # Reject anything that isn't a whitelisted image type
        raise HTTPException(400, "Only png, jpg, gif and webp images are allowed")

    # Generate a random filename
    filename = f"image-{uuid4()}{ext}"

    upload_dir = pathlib.Path(f"uploads-{session_id}")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Read the uploaded bytes and write them to disk under that name
    (upload_dir / filename).write_bytes(await file.read())

    # Build the public URL; request.base_url is e.g. "http://localhost:8000/"
    return {"url": f"{request.base_url}files/{filename}"}

@app.post("/call_agent")
def call_agent(session_id: str, text_query: str, image_url: str = None):
    '''
    Calls a local Gemma agent using Hugging Face pipelines and LangChain, to answer the users question
    '''

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": text_query},
        ],
    }]
 
    if image_url:
        filename = image_url.split("/files/")[-1]
        image_path = pathlib.Path(f"uploads-{session_id}") / filename
        b64_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        messages[0]["content"].append({
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{b64_image}",
        })

    config = {"configurable": {"thread_id": session_id}}

    result = agent.invoke(
        {"messages": messages},
        config=config
    )

    text_output = result["messages"][-1].content
    
    # Convert agent text response to speech
    speech = tts(text_output)
    audio_array = (speech["audio"][0] * 32767).astype(np.int16)

    wav_buffer = io.BytesIO()
    scipy.io.wavfile.write(wav_buffer, rate=speech["sampling_rate"], data=audio_array)
    audio_b64 = base64.b64encode(wav_buffer.getvalue()).decode("utf-8")

    return {
        "text": text_output,
        "audio_base64": f"data:audio/wav;base64,{audio_b64}",
    }

@app.delete("/end_session")
def end_session(session_id: str):
    '''
    Ends the session and deletes the directory with the uploaded files from a session
    '''
    shutil.rmtree(f"uploads-{session_id}", ignore_errors=True)

    return "Upload directory deleted successfully and session reset"


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")