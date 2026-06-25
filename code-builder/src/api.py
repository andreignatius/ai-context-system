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
    # optional carry-forward for a HUMAN fix (all empty = a fresh build):
    spec: str = ""
    tests: str = ""
    code: str = ""
    test_result: dict = {}
    fix_target: str = ""          # "" fresh | "spec" | "tests" | "code"
    feedback: str = ""

class BuildResponse(BaseModel):
    status: str
    spec: str
    code: str
    tests: str
    test_result: dict             # so the UI can carry it back on a fix

@app.post("/build", response_model=BuildResponse)
def build(req: BuildRequest):
    state = {"request": req.request, "feedback": req.feedback, "fix_target": req.fix_target}
    if req.fix_target:            # a human fix -> carry the prior artifacts forward (memory)
        state.update({"spec": req.spec, "tests": req.tests,
                      "code": req.code, "test_result": req.test_result})
    result = _graph.invoke(state, config={"callbacks": [_handler]})
    return BuildResponse(
        status=result["status"], spec=result["spec"], code=result["code"],
        tests=result["tests"], test_result=result.get("test_result", {}),
    )
