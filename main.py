from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from typing import Any


load_dotenv()  

# --- A2A PROTOCOL: AGENT CARD DEFINITION ---
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
# ---------------------------------------------

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str] 
    tools_used: list[str]

class Query(BaseModel):
    query: str
    
llm = ChatGroq(model_name="openai/gpt-oss-20b", 
                 temperature=0,
                  api_key=os.getenv("groq_api_key")  
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [ 
        (
            "system",
            """
            You are an AI research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instuctions}
            """
        ),
        ("human", "{query}"),
        ("placeholder", "{chat_history}"),
    ]
).partial(format_instuctions=parser.get_format_instructions())


#tools = [search_tool, wiki_tool, save_tool]
tools =[]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#query = input("What can i help you research? ")
#raw_response = agent_executor.invoke({"query": query})


app = FastAPI(title="AI Research Assistant Agent")

@app.get("/.well-known/agent.json")
async def get_agent_card():
    return JSONResponse(content=AGENT_CARD_DATA)


@app.post("/")
async def handle_rpc_request(request: Request):
  body = await request.json()

  method = body.get('method')

  if method == "message/send":
    #route to function handling that method
    result = await execute_research_logic(request)


  else: 
    #throw return an error if method doesnt exist
    error = {
      "jsonrpc": "2.0", 
      "id": body.get("id", None),
      "error": {
        "code": -32601, 
        "message": "Method not found"
      } 
    }

    return error

  return result

async def execute_research_logic(request: Request):
    
    body = await request.json()
    params = body.get("params", {}) if isinstance(body, dict) else {}

    # Try several common locations for the query text (Telex A2A shapes + fallbacks)
    query_text = None
    msg = params.get("message") or params.get("msg") or {}
    if isinstance(msg, dict):
        content = msg.get("content") or {}
        if isinstance(content, dict):
            query_text = content.get("text") or content.get("body") or content.get("content")
        if not query_text:
            query_text = msg.get("text") or msg.get("body")
    if not query_text:
        query_text = params.get("query") or params.get("text")
    if not query_text:
        query_text = body.get("query") or body.get("text") or body.get("message")

    if not query_text:
        return JSONResponse(status_code=400, content={"error": "No query text found in request"})

    try:
        raw_response = agent_executor.ainvoke({"query": query_text})

        def _extract_text(resp: Any) -> str:
            if isinstance(resp, dict):
                out = resp.get("output")
                if isinstance(out, list) and out:
                    first = out[0]
                    if isinstance(first, dict):
                        return first.get("text") or first.get("output_text") or str(first)
                    return str(first)
                return resp.get("output_text") or resp.get("text") or str(resp)
            if hasattr(resp, "output"):
                try:
                    o = resp.output
                    if isinstance(o, list) and o:
                        first = o[0]
                        if isinstance(first, dict):
                            return first.get("text") or first.get("output_text") or str(first)
                        return str(first)
                except Exception:
                    pass
            return str(resp)

        output_text = _extract_text(raw_response)

        # Try structured parse, fall back to raw text
        try:
            structured = parser.parse(output_text)
            result_payload = {"structured": structured.dict()}
        except Exception:
            result_payload = {"text": output_text}

        # If user asked to save and file was produced, return file
        file_path = Path("research_output.txt")
        if "save" in query_text.lower() and file_path.exists():
            return FileResponse(path=str(file_path), filename="research_output.txt", media_type="text/plain")

        # Return JSON-RPC style response if request used that format
        rpc_id = body.get("id")
        rpc_response = {"jsonrpc": "2.0", "id": rpc_id, "result": result_payload}
        return JSONResponse(content=rpc_response)
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")
   