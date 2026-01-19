from fastapi import FastAPI
from pydantic import BaseModel
from .handler import lambda_handler

app = FastAPI()

class AskBody(BaseModel):
    text: str
    filters: dict | None = None
    style: str | None = None

@app.post("/ask")
def ask(body: AskBody):
    event = {"text": body.text, "filters": body.filters or {}, "style": body.style}
    return lambda_handler(event, None)
