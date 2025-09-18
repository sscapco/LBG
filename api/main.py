from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Any, Dict, Optional

# reuse your existing orchestration entry
from src.orchestration.graph_runtime import run_conversation
# optional: for /v1/agents sanity
from src.orchestration.registry import load_registry

app = FastAPI(title="Governance Orchestrator API", version="0.1.0")

class ActRequest(BaseModel):
    session_id: str
    io_mode: str = "chat"
    message: Optional[str] = None      # for chat
    inputs: Optional[Dict[str, Any]] = None  # for non-chat IOs later
    selector: Optional[str] = None     # optional label filter (future)
    debug: bool = False

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/v1/agents")
def list_agents():
    return {"agents": [a.name for a in load_registry()]}

@app.post("/v1/act")
def act(req: ActRequest = Body(...)):
    # today we only handle chat (message). Later, pass req.inputs to specific IO agents.
    user_msg = req.message or ""
    envelope = run_conversation(
        session_id=req.session_id,
        user_message=user_msg,
        io_mode=req.io_mode
    )
    return envelope
