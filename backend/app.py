from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()
from backend.config import TAVILY_API_KEY 
from backend.agent import run


app = FastAPI(title="LangGraph RAG Agent API")


print("TAVILY_API_KEY present?", bool(TAVILY_API_KEY), "len=", len(TAVILY_API_KEY))
class QuestionRequest(BaseModel):
    question: str
    max_steps: int = 6

@app.post("/ask")
def ask_agent(req: QuestionRequest):
    try:
        answer = run(req.question, max_steps=req.max_steps)
        return {"question": req.question, "answer": answer}
    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting API Server on http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
