from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
from fastapi import FastAPI, HTTPException
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
        ("placeholder", "{chat_history}"),
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

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#query = input("What can i help you research? ")
#raw_response = agent_executor.invoke({"query": query})


app = FastAPI(title="AI Research Assistant Agent")

@app.get("/.well-known/agent-card.json")
async def get_agent_card():
    return JSONResponse(content=AGENT_CARD_DATA)



async def execute_research_logic(query_text: str):
    try:
        raw_response = agent_executor.invoke({"query": query.query})
        output = raw_response.get("output")

        if isinstance(output, str):
            output_text = output
        else:
            output_text = str(output)

        #  Check if the agent saved a file
        file_path = Path("research_output.txt")
        if "save" in query_text.lower() and file_path.exists():
            # Return the file directly
            return FileResponse(
                path=file_path,
                filename="research_output.txt",
                media_type="text/plain"
            )

        # Otherwise, return text
        return {"response": output_text}

    except Exception as e:
        print("Error in /research:", e)
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")
    

@app.post("/")
async def handle_a2a_message(message: dict[str, Any]):
    """
    The A2A standard entry point for receiving messages, wrapped in JSON-RPC 2.0.
    """
    request_id = message.get("id")
    
    try:
        # 1. Extract the query from the likely JSON-RPC structure (Telex)
        # We check for a common Telex structure: params -> input -> user_prompt
        query_text = message.get("params", {}).get("input", {}).get("user_prompt")
        
        # Fallback to check for a simpler structure if the above fails
        if not query_text:
             query_text = message.get("query") or message.get("payload")

        if not query_text:
            # Return a standardized JSON-RPC error for a missing parameter
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {"code": -32602, "message": "Invalid params: Missing 'user_prompt' in message payload."},
                    "id": request_id
                }
            )
        
        # 2. Execute the core research logic
        # execute_research_logic handles the agent run and FileResponse logic internally
        research_result = await execute_research_logic(query_text)
        
        # 3. Wrap the successful response in JSON-RPC 2.0 format
        # If research_result is a FileResponse, you'll need to handle it differently, 
        # but typically A2A agents return JSON. Assuming the core logic returns JSON:
        return JSONResponse(content={
            "jsonrpc": "2.0",
            # The result field contains the output of the agent execution
            "result": research_result, 
            "id": request_id
        })

    except HTTPException as e:
        # Re-raise explicit HTTP exceptions (e.g., from the core logic)
        raise e
        
    except Exception as e:
        print("Error during A2A message handling:", e)
        # 4. Handle generic errors with a JSON-RPC error structure
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Internal server error: {e}"},
                "id": request_id
            }
        )