from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Any, Literal, Optional, List, Dict
from uuid import uuid4
from datetime import datetime
import re
import asyncio 

load_dotenv() 

# --- 1. A2A PROTOCOL: PYDANTIC MODELS (Required for TaskResult structure) ---

class MessagePart(BaseModel):
    kind: Literal["text", "data", "file"]
    text: Optional[str] = None
    data: Optional[Any] = None
    file_url: Optional[str] = None

class A2AMessage(BaseModel):
    kind: Literal["message"] = "message"
    role: Literal["user", "agent", "system"]
    parts: List[MessagePart]
    messageId: str = Field(default_factory=lambda: str(uuid4()))
    taskId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TaskStatus(BaseModel):
    state: Literal["working", "completed", "input-required", "failed"]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    message: Optional[A2AMessage] = None

class Artifact(BaseModel):
    artifactId: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    parts: List[MessagePart]

class TaskResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    contextId: str = Field(default_factory=lambda: str(uuid4()))
    status: TaskStatus
    artifacts: List[Artifact] = []
    history: List[A2AMessage] = []
    kind: Literal["task"] = "task"

class JSONRPCResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str
    result: Optional[TaskResult] = None
    error: Optional[Dict[str, Any]] = None

# --- 2. AGENT CARD DEFINITION (Fixed path and compliance) ---
AGENT_CARD_DATA = {
    "name": "AI Research Assistant Agent",
    "description": "An AI agent that performs research using search, Wikipedia, and can save results.",
    "version": "1.0",
    "service_url": "https://web-production-b95ea.up.railway.app/", 
    "protocols": ["a2a-messaging-1.0"],
    "interfaces": [
        {"name": "research", "description": "Performs in-depth research on a given topic."},
    ],
    "contact": "ndudimichael@gmail.com", 
    "public_key": None 
}

# --- 3. LANGCHAIN SETUP (Latency and prompt fixed) ---
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str] 
    tools_used: list[str]
    
# FIX: Use a lower-latency model for better A2A performance (llama3-8b is one of the fastest)
llm = ChatGroq(
    model_name="llama3-8b-8192", 
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY") 
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [ 
        (
            "system",
            """
            You are an AI research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            When the user asks to 'save' the results, use the save_tool and then include the file content in your final response.
            Wrap the output in this format and provide no other text\n{format_instuctions}
            """
        ),
        # FIX: Removed "{chat_history}" to ensure stateless A2A requests
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instuctions=parser.get_format_instructions())


tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# FIX: Ensure agent_executor is using the asynchronous version correctly
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


app = FastAPI(title="AI Research Assistant Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIX: Corrected Agent Card path
@app.get("/.well-known/agent-card.json")
async def get_agent_card():
    return JSONResponse(content=AGENT_CARD_DATA)

# --- 4. CORE A2A ENDPOINT (Adapted to TaskResult structure) ---

@app.post("/")
# NOTE: The working agent used a specific path like "/a2a/nasa". 
# If this fails, change the decorator to @app.post("/a2a/research") and update your Telex URL.
async def handle_rpc_request(request: Request):
    try:
        body = await request.json()
        request_id = body.get("id", str(uuid4()))
        method = body.get('method')
        
        if method == "message/send":
            # 1. Extract the query using the robust logic
            query_text = extract_user_message_from_request(body)
            
            # 2. Execute the LangChain agent logic
            agent_result_payload = await execute_research_logic(query_text)
            
            # 3. Format the result into the complex TaskResult structure
            task_result = await create_task_result(agent_result_payload, request_id, body)
            
            # 4. Final JSON-RPC response
            rpc_response = JSONRPCResponse(id=request_id, result=task_result)
            return rpc_response.model_dump(by_alias=True)

        else: 
            error = {
              "jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32601, "message": "Method not found"}
            } 
            return JSONResponse(status_code=400, content=error)
    
    except HTTPException as e:
        # Catch internal errors from execute_research_logic and format as JSON-RPC error
        error_response = {
            "jsonrpc": "2.0", "id": body.get("id", None),
            "error": {"code": -32603, "message": e.detail}
        }
        return JSONResponse(status_code=e.status_code, content=error_response)
    except Exception as e:
        error_response = {
            "jsonrpc": "2.0", "id": body.get("id", None),
            "error": {"code": -32603, "message": f"Unexpected error: {e}"}
        }
        return JSONResponse(status_code=500, content=error_response)


# --- 5. HELPER FUNCTIONS ---

def extract_user_message_from_request(body: dict) -> str:
    """Extract user message from the deeply nested A2A request body, adapting from the working agent's logic."""
    try:
        params = body.get('params', {})
        # Look for the current message text
        msg = params.get('message', {})
        
        # Check parts array which often contains the user's input message
        parts = msg.get('parts', [])
        for part in parts:
            if part.get('kind') == 'text' and part.get('text'):
                # Extract clean command, stripping potential HTML formatting from the Telex chat UI
                text = part['text'].strip()
                if text and text.startswith('<p>') and text.endswith('</p>'):
                    clean_text = re.sub('<[^<]+?>', '', text).strip()
                    return clean_text
                return text

    except Exception as e:
        print(f"Error extracting message: {e}")
        # Default to a generic query if extraction fails
        return "summarize a general topic" 

async def execute_research_logic(query_text: str):
    """
    Runs the LangChain agent asynchronously and handles file I/O, 
    returning a dictionary payload instead of a JSONResponse.
    """
    file_path = Path("research_output.txt")
    if file_path.exists():
        file_path.unlink() # Cleanup before run

    try:
        # FIX: Ensure we await the ainvoke call!
        raw_response = await agent_executor.ainvoke({"query": query_text}) 
        output_text = raw_response.get("output", str(raw_response))

        result_payload = {"text": output_text}

        # FIX: Handle file functionality by reading content and embedding it
        if "save" in query_text.lower() and file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            file_path.unlink() # Clean up the ephemeral file
            
            # Embed file content into the payload
            result_payload["file_content"] = file_content
            result_payload["file_note"] = "The content of the requested saved file is embedded below."
            
        return result_payload
    
    except Exception as e:
        print(f"Agent execution failed: {e}")
        # Throw an HTTPException to be caught in the main endpoint
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}")

async def create_task_result(agent_payload: dict, request_id: str, original_body: dict) -> TaskResult:
    """Creates the complex TaskResult object expected by Telex."""
    
    # 1. Determine the final response text
    response_text = agent_payload.get("text", "No response generated.")
    if agent_payload.get("file_content"):
        response_text += (
            f"\n\n--- SAVED FILE CONTENT ---\n"
            f"{agent_payload['file_content']}"
            f"\n--------------------------"
        )
        
    # 2. Create the final A2AMessage
    response_message = A2AMessage(
        role="agent",
        parts=[
            MessagePart(kind="text", text=response_text)
        ],
        messageId=str(uuid4()),
        taskId=str(uuid4()), 
    )

    # 3. Create Artifacts
    artifacts = [
        Artifact(
            artifactId=str(uuid4()),
            name="research_output",
            parts=[MessagePart(kind="text", text=response_text)]
        )
    ]
    
    # 4. Build TaskResult (simplified history for now)
    history = [response_message]
    
    result = TaskResult(
        id=str(uuid4()),
        contextId=str(uuid4()),
        status=TaskStatus(
            state="completed", # Crucial: Must be completed upon successful execution
            timestamp=datetime.utcnow().isoformat() + "Z",
            message=response_message
        ),
        artifacts=artifacts,
        history=history,
        kind="task"
    )
    return result

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    # Note: For Railway, this run block might not be used if it uses a Procfile/ASGI server config
    uvicorn.run(app, host="0.0.0.0", port=port)