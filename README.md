# Research Agent

An AI research assistant powered by FastAPI, LangChain, and Groq, designed to conduct automated research on any topic and optionally save the results to a text file for download.

---

## 🚀 Features

💬 Accepts natural language queries like “What is quantum computing?”

🧩 Uses LangChain and Groq LLM for intelligent research synthesis

💾 “Save to file” capability — just include “save” in your query, e.g.

“Research about CRISPR technology and save to file”
This automatically generates research_output.txt and returns it as a downloadable file.

⚡ Built with FastAPI for high-performance async API calls

☁️ Easily deployable (tested on Railway)




---

## Tech Stack

Backend Framework: FastAPI

LLM Framework: LangChain

LLM Provider: Groq

Language: Python 3.11+



```

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/mik3lson/research_agent
cd research agent

```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Running the Server

### Start the FastAPI development server:
```bash
uvicorn app:app --reload
```
### You should see output similar to
```arduino
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## 🧩 API Endpoints
POST /research

Description:
Conducts research on a given topic and returns either a text response or a downloadable file (if “save” is included in the query).

Request Body:
```json

{
  "query": "Explain the greenhouse effect and save to file"
}
```

Response (JSON Example):
```json
{
  "response": "The greenhouse effect is a process that warms the Earth's surface..."
}
```

Response (File Example):
If the query includes “save,” a .txt file (e.g., research_output.txt) will be returned instead.

---


## 🧪 Testing
###You can test the API using:
curl<br>
```curl
curl -X POST https://https://web-production-b95ea.up.railway.app/research \
-H "Content-Type: application/json" \
-d '{"query": "What is K-pop, save to file"}'
```
Postman<br>
http://127.0.0.1:8000/docs— FastAPI’s built-in Swagger UI


# Alternatively the api is hosted here:
```link
https://web-production-b95ea.up.railway.app/
```
