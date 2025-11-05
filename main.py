from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path


load_dotenv()  


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

@app.post("/research")
async def run_research(query: Query):
    try:
        raw_response = agent_executor.invoke({"query": query.query})
        output = raw_response.get("output")

        if isinstance(output, str):
            output_text = output
        else:
            output_text = str(output)

        # 🔍 Check if the agent saved a file
        file_path = Path("research_output.txt")
        if "save" in query.query.lower() and file_path.exists():
            # ✅ Return the file directly
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