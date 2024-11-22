from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
from transformers import pipeline
from pydantic import BaseModel

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
templates = Jinja2Templates(directory="templates")

app = FastAPI()

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 130
@app.get("/", response_class=HTMLResponse)
def render_home(request: Request):
    """
    Renders the home page in html format

    :param request:

    :return: return the html file
    """
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/summarize")
async def summarize(request: SummarizeRequest):
    """
    Summarize input text

    :param request: Get the request body as SummarizeRequest object
    :return: return the summary in SummarizeRequest format
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        if request.max_length < 30 or request.max_length > 500:
            raise HTTPException(status_code=400, detail="Max length must be between 30 and 500")

        summary = summarizer(
            request.text,
            max_length=request.max_length,
            min_length=30,
            do_sample=False
        )

        return {"summary": summary[0]["summary_text"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))