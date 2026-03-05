from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from backend.agent import run

load_dotenv()
app = FastAPI(title="LangGraph RAG Agent API")

class QuestionRequest(BaseModel):
    question: str
    max_steps: int = 6

@app.post("/ask")
def ask_agent(req: QuestionRequest):
    try:
        answer = run(req.question, max_steps=req.max_steps)
        return {"question": req.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting API Server on http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
