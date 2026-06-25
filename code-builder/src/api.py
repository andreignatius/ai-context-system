"""FastAPI wrapper: expose the code-builder as an HTTP service (M11).
    uvicorn src.api:app --reload
"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import build_graph

app = FastAPI(title="Code Builder")
_graph = build_graph()                      # built ONCE at startup, reused per request


class BuildRequest(BaseModel):
    request: str


class BuildResponse(BaseModel):
    status: str                             # "ok" if it converged autonomously, else "failed"
    spec: str
    code: str
    tests: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/build", response_model=BuildResponse)
def build(req: BuildRequest):
    result = _graph.invoke({"request": req.request})   # one-shot, no human; the judge self-heals
    return BuildResponse(
        status=result["status"],
        spec=result["spec"],
        code=result["code"],
        tests=result["tests"],
    )
