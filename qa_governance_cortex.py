from agents import (
    Agent,
    Runner,
    function_tool,
    OpenAIChatCompletionsModel,
    set_default_openai_client,
    set_tracing_disabled,
)
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
import asyncio
from dotenv import load_dotenv
from datetime import datetime
import os
import json
import pandas as pd
import numpy as np
import re

from automation_registry2 import (
    get_automation_info,
    run_automation_step,
    format_automation_result_for_user,
    list_available_automations
)

# Import LLM configuration 
from llm_config import initialize_clients, DEPLOYMENT_NAME, EMBEDDING_MODEL

######### Environment & Clients #########

load_dotenv()

# Initialize clients using the factory
client, async_client = initialize_clients()

# Set up for OpenAI Agents SDK
set_default_openai_client(async_client)
set_tracing_disabled(disabled=True)

shared_model = OpenAIChatCompletionsModel(
    model=DEPLOYMENT_NAME,
    openai_client=async_client,
)

######### Governance Data Loading #########

EXCEL_PATH = "nodes_edges_governance.xlsx"

nodes_raw_df = pd.read_excel(EXCEL_PATH, sheet_name="Nodes")
edges_raw_df = pd.read_excel(EXCEL_PATH, sheet_name="Edges")

# Normalise Nodes columns
nodes_df = nodes_raw_df.rename(
    columns={
        "Stage Name": "Stage_Name",
        "Step ID": "Step_ID",
        "Step name": "Step_Name",
        "Purpose (Brief one liner to outline purpose of the task)": "Purpose",
    }
)

# Ensure Automatable + Automation_step exist even if not yet in sheet
if "Automatable" not in nodes_df.columns:
    nodes_df["Automatable"] = False
if "Automation_step" not in nodes_df.columns:
    nodes_df["Automation_step"] = ""

# Normalise Edges columns
edges_df = edges_raw_df.rename(
    columns={
        "Question/Probe": "Question_Probe",
        "Transactional vs Analytical": "Transactional_vs_Analytical",
    }
)

required_node_cols = ["Stage", "Stage_Name", "Step_ID", "Step_Name", "Purpose", "Description"]
missing_node_cols = [c for c in required_node_cols if c not in nodes_df.columns]
if missing_node_cols:
    raise ValueError(f"Missing expected columns in Nodes sheet: {missing_node_cols}")

required_edge_cols = ["from", "to", "edge_type", "guard_condition", "Question_Probe"]
missing_edge_cols = [c for c in required_edge_cols if c not in edges_df.columns]
if missing_edge_cols:
    raise ValueError(f"Missing expected columns in Edges sheet: {missing_edge_cols}")

# Normalise Step_IDs and edge endpoints
nodes_df["Step_ID"] = nodes_df["Step_ID"].astype(str).str.strip().str.upper()
edges_df["from"] = edges_df["from"].astype(str).str.strip().str.upper()
edges_df["to"] = edges_df["to"].astype(str).str.strip().str.upper()

# Normalise Automatable & Automation_step
nodes_df["Automatable"] = (
    nodes_df["Automatable"]
    .astype(str)
    .str.strip()
    .str.upper()
    .isin(["TRUE", "YES", "Y", "1"])
)

nodes_df["Automation_step"] = nodes_df["Automation_step"].astype(str).str.strip()

# Canonical workflow order: row order in Nodes
ORDERED_STEP_IDS: List[str] = list(nodes_df["Step_ID"])

# Position of step in canonical order (or large number if unknown)
def step_index(step_id: str) -> int:
    try:
        return ORDERED_STEP_IDS.index(step_id)
    except ValueError:
        return 10_000

# Return metadata for a given step_id 
def get_step_record(step_id: str) -> Optional[Dict]:
    row = nodes_df[nodes_df["Step_ID"] == step_id]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "id": r["Step_ID"],
        "name": r["Step_Name"],
        "purpose": r["Purpose"],
        "description": r["Description"],
        "stage": r["Stage"],
        "stage_name": r["Stage_Name"],
        "automatable": bool(r.get("Automatable", False)),
        "automation_step": (r.get("Automation_step") or "").strip() or None,
    }


######### Embedding Utilities #########

def get_embedding(text: str) -> List[float]:
    text = (text or "").replace("\n", " ")
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


EMBED_CACHE_PATH = "step_embeddings_cache.json"
STEP_EMBEDDINGS: Dict[str, Dict] = {}

if os.path.exists(EMBED_CACHE_PATH):
    try:
        with open(EMBED_CACHE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for sid, data in raw.items():
            STEP_EMBEDDINGS[str(sid).upper()] = {
                "embedding": data["embedding"],
                "text": data.get("text", ""),
            }
        print(f"✅ Loaded {len(STEP_EMBEDDINGS)} step embeddings from cache")
    except Exception as e:
        print(f"⚠️ Failed to load embeddings cache: {e}. Recomputing...")
        STEP_EMBEDDINGS = {}
else:
    print("ℹ️ No embeddings cache found. Computing step embeddings...")


if not STEP_EMBEDDINGS:
    for _, row in nodes_df.iterrows():
        sid = str(row["Step_ID"]).strip().upper()
        text_parts = [
            str(row["Stage_Name"]),
            str(row["Step_Name"]),
            str(row["Purpose"]),
            str(row["Description"]),
        ]
        combined_text = ". ".join([p for p in text_parts if p and p != "nan"])
        emb = get_embedding(combined_text)
        STEP_EMBEDDINGS[sid] = {
            "embedding": emb,
            "text": combined_text,
        }

    with open(EMBED_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(STEP_EMBEDDINGS, f)
    print(f"✅ Precomputed and cached embeddings for {len(STEP_EMBEDDINGS)} steps")


######### Alias-based deterministic mapping #########

STEP_ALIASES: Dict[str, List[str]] = {}

for _, row in nodes_df.iterrows():
    sid = str(row["Step_ID"]).strip().upper()
    name = str(row["Step_Name"]).strip()
    aliases = set()

    if sid:
        aliases.add(sid.lower())

    if name:
        aliases.add(name.lower())

        # contents inside parentheses
        for match in re.findall(r"\(([^)]+)\)", name):
            alias = match.strip()
            if alias:
                aliases.add(alias.lower())

        # acronym from uppercase letters
        caps = "".join(ch for ch in name if ch.isupper())
        if len(caps) >= 3:
            aliases.add(caps.lower())

    STEP_ALIASES[sid] = sorted(aliases)

# Map natural-language text to a Step_ID using id, alias, substring, then embeddings
def map_text_to_step_id(text: str, emb_threshold: float = 0.5) -> Tuple[Optional[str], str, float]:
    if not text:
        return None, "none", 0.0

    t = text.strip().lower()
    if not t:
        return None, "none", 0.0

    # 1) Direct S-identifier
    m = re.search(r"\b(s\d{1,3})\b", t)
    if m:
        sid = m.group(1).upper()
        if sid in STEP_EMBEDDINGS:
            return sid, "id_match", 1.0

    # 2) Alias match
    for sid, aliases in STEP_ALIASES.items():
        for alias in aliases:
            if not alias:
                continue
            if re.search(r"\b" + re.escape(alias) + r"\b", t) or alias in t:
                return sid, "alias_match", 1.0

    # 3) Substring match on step names
    best_sid = None
    best_len = 0
    for _, row in nodes_df.iterrows():
        sid = row["Step_ID"]
        name = str(row["Step_Name"]).lower()
        if name and name in t:
            if len(name) > best_len:
                best_len = len(name)
                best_sid = sid
    if best_sid:
        return best_sid, "substring_match", 1.0

    # 4) Embeddings fallback
    action_emb = get_embedding(text)
    best_step = None
    best_score = 0.0
    for sid, data in STEP_EMBEDDINGS.items():
        score = cosine_similarity(action_emb, data["embedding"])
        if score > best_score:
            best_score = score
            best_step = sid
    if best_step and best_score >= emb_threshold:
        return best_step, "embedding", best_score

    return None, "none", best_score


######### Session memory #########

class GovernanceSessionState(BaseModel):
    last_intent: Optional[str] = None
    last_focus_step_id: Optional[str] = None
    last_anchor_step_id: Optional[str] = None
    last_next_step_ids: List[str] = Field(default_factory=list)
    last_completed_ids: List[str] = Field(default_factory=list)
    last_in_progress_ids: List[str] = Field(default_factory=list)
    last_referenced_ids: List[str] = Field(default_factory=list)
    last_automatable_step_id: Optional[str] = None
    last_automation_step: Optional[str] = None
    timestamp: Optional[str] = None


SESSION_STATE: Dict[str, Dict] = {}


######### Structured models for parsing & workflow reasoning #########

class ParsedAction(BaseModel):
    description: str
    status: str  # "completed" | "in_progress" | "mentioned_only"


class IntentParseResult(BaseModel):
    intent: str
    actions: List[ParsedAction]
    focus_action_index: Optional[int]
    is_starting_question: bool = False


class StepCandidate(BaseModel):
    step_id: str
    score: float


class MappedSteps(BaseModel):
    intent: str
    is_starting_question: bool
    completed_ids: List[str]
    in_progress_ids: List[str]
    referenced_ids: List[str]
    focus_step_candidate_id: Optional[str]
    semantic_candidates: List[StepCandidate] = Field(default_factory=list)
    has_direct_mapping: bool = False  # any id/alias/substring match found


class WorkflowState(BaseModel):
    intent: str
    is_starting_question: bool
    completed_ids: List[str]
    in_progress_ids: List[str]
    referenced_ids: List[str]
    anchor_step_id: Optional[str]
    focus_step_id: Optional[str]
    next_step_ids: List[str]
    missing_prereq_ids: List[str]
    step_details: Dict[str, Dict]
    prereq_edges: Dict[str, List[Dict]]
    semantic_candidates: List[StepCandidate]
    needs_disambiguation: bool


class GovernanceToolResult(BaseModel):
    answer: str
    updated_state: GovernanceSessionState


######### LLM parsing: intent + actions #########

def parse_intent_and_actions(
    user_message: str,
    previous_state: Optional[GovernanceSessionState],
) -> IntentParseResult:
    previous_state_safe = json.loads(previous_state.model_dump_json()) if previous_state else {}

    system_prompt = (
        "You are a classifier for a data governance workflow. "
        "You do NOT guess step IDs or names; you ONLY work with natural-language actions and intents."
    )

    user_prompt = f"""
User message:
\"\"\"{user_message}\"\"\"

Previous workflow state (may be empty):
{json.dumps(previous_state_safe, indent=2)}

Your tasks:

1) Determine the user's intent:
   - "what_next"       
       → they clearly ask what to do next or what comes after something 
         (e.g. "what should I do next?", "what comes after DPWG?").
   - "help_current"    
       → they ask for help completing something they say they are currently doing
         (e.g. "I'm working on the DOI form, what do I need to include?").
   - "ask_about_step"  
       → they want to understand what a specific step/process is or involves
         (e.g. "What does the DPWG process involve?",
                "Do we have to register our product somewhere so people can discover it?",
                "We might touch customer data—who handles privacy and what form do we fill in?").
   - "what_missed"     
       → they ask which steps they might have missed or skipped
         (e.g. "What governance steps might I have missed?").
   - "run_automation"  
       → they explicitly ask you to RUN an automation you (or the system)
         have already offered for a step (e.g. "yes, run the naming check",
         "please validate the name", "run the automation now").
   - "other"           
       → anything else.

   IMPORTANT:
   - If the message is a general conceptual question ("Do we need to register X?",
     "Who handles privacy?", "Do we need a go/no-go step?"), but they do NOT clearly talk
     about sequencing ("next", "after", "then"), choose intent = "ask_about_step".
   - Only choose intent = "what_next" when the user explicitly frames it as the next step
     in a sequence (e.g. "what comes after DPWG", "what do I do next", "what's the next step").
   - Questions like "Where do I start?" or "How do I get started?" are intent = "what_next"
     but are also STARTING QUESTIONS (see is_starting_question below).

2) Extract ACTIONS mentioned in the message as natural language phrases.
   For each action, assign a status:
     - "completed"      
          → they say they finished/endorsed/approved it 
     - "in_progress"    
          → they say they are doing / preparing / working on it now
     - "mentioned_only" 
          → they refer to it but don't say if it's done or in progress
            

3) Choose focus_action_index:
   - If the message clearly refers to one main step or action, set focus_action_index
     to the zero-based index in the actions list.
   - If there is no clear main action, set focus_action_index to null.

4) Decide if this is a STARTING QUESTION:
   - is_starting_question = true if the user is essentially asking:
       * "where do I start?"
       * "what's the first thing I should do?"
       * "how do I get started with this new data thing?"
     and they have NOT described any concrete progress or completed steps yet.
   - Otherwise, is_starting_question = false.

5) For vague follow-ups like:
     "yes, more details"
     "run the check"
     "please automate this"
   use the previous_state to infer context:
   - If the user seems to be saying "yes, run the automation" or similar,
     and previous_state contains a last_automatable_step_id, then choose
     intent = "run_automation".

OUTPUT FORMAT (strict JSON only):

{{
  "intent": "what_next" | "help_current" | "ask_about_step" | "what_missed" | "run_automation" | "other",
  "actions": [
    {{
      "description": "<short phrase>",
      "status": "completed" | "in_progress" | "mentioned_only"
    }}
  ],
  "focus_action_index": <integer index or null>,
  "is_starting_question": true | false
}}
    """

    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=700,
    )

    try:
        raw = json.loads(resp.choices[0].message.content)
    except Exception:
        raw = {}

    intent = raw.get("intent", "other")
    if intent not in ["what_next", "help_current", "ask_about_step", "what_missed", "run_automation", "other"]:
        intent = "other"

    actions_raw = raw.get("actions", [])
    actions: List[ParsedAction] = []
    for a in actions_raw:
        if not isinstance(a, dict):
            continue
        desc = (a.get("description") or "").strip()
        status = (a.get("status") or "mentioned_only").strip()
        if not desc:
            continue
        if status not in ["completed", "in_progress", "mentioned_only"]:
            status = "mentioned_only"
        actions.append(ParsedAction(description=desc, status=status))

    focus_idx = raw.get("focus_action_index")
    if not isinstance(focus_idx, int) or not (0 <= focus_idx < len(actions)):
        focus_idx = None

    is_starting = bool(raw.get("is_starting_question", False))
    lower_msg = (user_message or "").lower()

    # Heuristic nudge for obvious starting questions
    starting_phrases = [
        "where do i start",
        "where should i start",
        "how do i start",
        "how do we start",
        "how do i get started",
        "how do we get started",
        "first thing i should do",
        "first thing we should do",
        "where to begin",
    ]
    if any(p in lower_msg for p in starting_phrases):
        is_starting = True

    # Extra rule-based override: strongly detect "what next" questions
    if any(
        phrase in lower_msg
        for phrase in [
            "what should i do next",
            "what do i do next",
            "what comes after",
            "what's after",
            "what is after",
            "what comes next",
            "next step",
            "what next",
        ]
    ):
        if intent not in ["what_missed", "run_automation"]:
            intent = "what_next"

    return IntentParseResult(
        intent=intent,
        actions=actions,
        focus_action_index=focus_idx,
        is_starting_question=is_starting,
    )


######### Map parsed actions to step IDs #########

def map_parsed_to_steps(
    user_message: str,
    parsed: IntentParseResult,
    previous_state: Optional[GovernanceSessionState],
) -> MappedSteps:
    intent = parsed.intent
    actions = parsed.actions
    focus_idx = parsed.focus_action_index
    is_starting = parsed.is_starting_question

    # If no actions but we have previous state and this is clearly a follow-up,
    # treat as referring to last focus/anchor step.
    if not actions and previous_state:
        last_focus = previous_state.last_focus_step_id or previous_state.last_anchor_step_id
        if last_focus in ORDERED_STEP_IDS:
            rec = get_step_record(last_focus)
            synthetic_desc = f"previous step {last_focus}: {rec['name']}" if rec else f"previous step {last_focus}"
            status = "completed" if intent == "what_next" else "mentioned_only"
            actions = [ParsedAction(description=synthetic_desc, status=status)]
            focus_idx = 0

    completed_ids: List[str] = []
    in_progress_ids: List[str] = []
    referenced_ids: List[str] = []
    focus_step_candidate_id: Optional[str] = None
    has_direct_mapping = False  # any id/alias/substring match found

    # 1) Map each action to a step (ID/alias/embedding)
    for idx, a in enumerate(actions):
        desc = a.description
        status = a.status
        step_id, method, score = map_text_to_step_id(desc)
        if not step_id:
            continue

        if method in ("id_match", "alias_match", "substring_match"):
            has_direct_mapping = True

        if status == "completed":
            completed_ids.append(step_id)
        elif status == "in_progress":
            in_progress_ids.append(step_id)

        if step_id not in referenced_ids:
            referenced_ids.append(step_id)

        if focus_idx is not None and idx == focus_idx:
            focus_step_candidate_id = step_id

    def unique(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    completed_ids = [s for s in unique(completed_ids) if s in ORDERED_STEP_IDS]
    in_progress_ids = [s for s in unique(in_progress_ids) if s in ORDERED_STEP_IDS]
    referenced_ids = [s for s in unique(referenced_ids) if s in ORDERED_STEP_IDS]

    # 2) Semantic candidates from the entire user message (for vague/general questions)
    semantic_candidates: List[StepCandidate] = []
    try:
        msg_emb = get_embedding(user_message)
        scores: List[Tuple[str, float]] = []
        for sid, data in STEP_EMBEDDINGS.items():
            score = cosine_similarity(msg_emb, data["embedding"])
            scores.append((sid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = scores[:3]
        semantic_candidates = [StepCandidate(step_id=sid, score=score) for sid, score in top_k]
    except Exception:
        semantic_candidates = []

    # 3) If we still have no referenced step at all, but have a reasonably strong semantic match,
    #    use the top candidate as a soft reference.
    if not referenced_ids and semantic_candidates:
        top = semantic_candidates[0]
        if top.score >= 0.30:
            referenced_ids.append(top.step_id)
            if focus_step_candidate_id is None:
                focus_step_candidate_id = top.step_id

    # 4) Fallback for vague follow-ups: no steps from actions nor semantic, but previous_state exists.
    if not referenced_ids and previous_state:
        last_focus = previous_state.last_focus_step_id or previous_state.last_anchor_step_id
        if last_focus in ORDERED_STEP_IDS:
            referenced_ids.append(last_focus)
            if intent == "what_next" and last_focus not in completed_ids:
                completed_ids.append(last_focus)
            focus_step_candidate_id = last_focus

    if focus_step_candidate_id not in ORDERED_STEP_IDS:
        focus_step_candidate_id = None

    return MappedSteps(
        intent=intent,
        is_starting_question=is_starting,
        completed_ids=completed_ids,
        in_progress_ids=in_progress_ids,
        referenced_ids=referenced_ids,
        focus_step_candidate_id=focus_step_candidate_id,
        semantic_candidates=semantic_candidates,
        has_direct_mapping=has_direct_mapping,
    )


######### Workflow reasoning on graph #########

def compute_workflow_state(mapped: MappedSteps) -> WorkflowState:
    intent = mapped.intent
    is_starting = mapped.is_starting_question
    completed_ids = mapped.completed_ids
    in_progress_ids = mapped.in_progress_ids
    referenced_ids = mapped.referenced_ids
    focus_candidate_id = mapped.focus_step_candidate_id
    semantic_candidates = mapped.semantic_candidates
    has_direct_mapping = mapped.has_direct_mapping

    def most_advanced(ids: List[str]) -> Optional[str]:
        return max(ids, key=step_index) if ids else None

    highest_completed = most_advanced(completed_ids)
    highest_in_progress = most_advanced(in_progress_ids)
    highest_referenced = most_advanced(referenced_ids)

    focus_step_id: Optional[str] = None
    anchor_step_id: Optional[str] = None

    #  Select focus & anchor based on intent 
    if intent == "ask_about_step":
        focus_step_id = focus_candidate_id or highest_referenced or highest_in_progress or highest_completed
        anchor_step_id = focus_step_id or highest_completed or highest_in_progress or highest_referenced

    elif intent == "help_current":
        focus_step_id = focus_candidate_id or highest_in_progress or highest_referenced or highest_completed
        anchor_step_id = focus_step_id or highest_completed or highest_in_progress or highest_referenced

    elif intent == "what_next":
        anchor_step_id = highest_completed or highest_in_progress or highest_referenced
        focus_step_id = None  # we typically focus on the next step

    elif intent == "what_missed":
        anchor_step_id = highest_completed or highest_in_progress or highest_referenced
        focus_step_id = anchor_step_id

    else:  # "other"
        focus_step_id = focus_candidate_id or highest_referenced or highest_in_progress or highest_completed
        anchor_step_id = focus_step_id or highest_completed or highest_in_progress or highest_referenced

    # Ambiguity detection using semantic candidates 
    needs_disambiguation = False
    if (
        len(semantic_candidates) >= 2
        and intent in ("ask_about_step", "help_current", "what_next")
        and not has_direct_mapping  # only use semantics when we don't have a strong explicit match
    ):
        c1, c2 = semantic_candidates[0], semantic_candidates[1]
        # ambiguous if both reasonably strong and fairly close
        if (
            c1.step_id != c2.step_id
            and c1.score >= 0.40   
            and c2.score >= 0.35
            and (c1.score - c2.score) <= 0.15   
        ):
            # If the user hasn't described explicit progress, treat as ambiguous conceptual question
            if not completed_ids and not in_progress_ids:
                needs_disambiguation = True

    #  Determine next steps 
    next_step_ids: List[str] = []

    if intent == "what_next" and not needs_disambiguation:
        # 1) Use graph edges from anchor_step_id, if present
        if anchor_step_id:
            outgoing = edges_df[
                (edges_df["from"] == anchor_step_id)
                & (edges_df["edge_type"].astype(str).str.lower() == "mandatory")
            ]
            for _, edge in outgoing.iterrows():
                to_step = edge["to"]
                if to_step in ORDERED_STEP_IDS and to_step not in completed_ids:
                    if to_step not in next_step_ids:
                        next_step_ids.append(to_step)

        # 2) If we have NO anchor and this IS a starting question → suggest the first step (or its semantic mapping)
        if not anchor_step_id and not next_step_ids and is_starting:
            if referenced_ids:
                candidate = referenced_ids[0]
            else:
                candidate = ORDERED_STEP_IDS[0] if ORDERED_STEP_IDS else None
            if candidate:
                next_step_ids = [candidate]
                focus_step_id = candidate

        # 3) If there is progress (completed) but graph gave nothing → canonical next after highest_completed
        if not next_step_ids and highest_completed:
            idx = step_index(highest_completed)
            if idx < len(ORDERED_STEP_IDS) - 1:
                candidate = ORDERED_STEP_IDS[idx + 1]
                if candidate not in completed_ids:
                    next_step_ids.append(candidate)

        # 4) If still no next_step_ids and there is a referenced step but no actual progress,
        #    treat as being about that referenced step (e.g. go/no-go, registration, privacy).
        if not next_step_ids and not completed_ids and highest_referenced and not is_starting:
            focus_step_id = focus_step_id or highest_referenced

        # 5) If we now have next_step_ids and no focus_step yet, focus on the next step
        if next_step_ids and not focus_step_id:
            focus_step_id = next_step_ids[0]

    # If ambiguous, don't commit to a single focus or next step.
    if needs_disambiguation:
        focus_step_id = None
        anchor_step_id = None
        next_step_ids = []

    #  Missing mandatory prerequisites 
    missing_prereq_ids: List[str] = []
    prereq_edges: Dict[str, List[Dict]] = {}

    targets = set(next_step_ids)
    if focus_step_id:
        targets.add(focus_step_id)

    for target in targets:
        incoming = edges_df[
            (edges_df["to"] == target)
            & (edges_df["edge_type"].astype(str).str.lower() == "mandatory")
        ]
        for _, edge in incoming.iterrows():
            from_step = edge["from"]
            if (
                from_step in ORDERED_STEP_IDS
                and from_step not in completed_ids
                and from_step not in in_progress_ids
            ):
                if from_step not in missing_prereq_ids:
                    missing_prereq_ids.append(from_step)
                prereq_edges.setdefault(target, []).append(
                    {
                        "from": from_step,
                        "question": edge.get("Question_Probe", None),
                        "guard_condition": edge.get("guard_condition", None),
                    }
                )

    #  Step details for all relevant steps 
    relevant_ids = set(
        completed_ids
        + in_progress_ids
        + referenced_ids
        + next_step_ids
        + missing_prereq_ids
    )
    if focus_step_id:
        relevant_ids.add(focus_step_id)
    if anchor_step_id:
        relevant_ids.add(anchor_step_id)
    for c in semantic_candidates:
        relevant_ids.add(c.step_id)

    step_details: Dict[str, Dict] = {}
    for s_id in relevant_ids:
        rec = get_step_record(s_id)
        if rec:
            step_details[s_id] = rec

    return WorkflowState(
        intent=intent,
        is_starting_question=is_starting,
        completed_ids=completed_ids,
        in_progress_ids=in_progress_ids,
        referenced_ids=referenced_ids,
        anchor_step_id=anchor_step_id,
        focus_step_id=focus_step_id,
        next_step_ids=next_step_ids,
        missing_prereq_ids=missing_prereq_ids,
        step_details=step_details,
        prereq_edges=prereq_edges,
        semantic_candidates=semantic_candidates,
        needs_disambiguation=needs_disambiguation,
    )


######### Tool: end-to-end analysis &  answer generation #########
# Main tool that the Agent calls.
@function_tool
def analyze_user_query(user_message: str, previous_state_json: Optional[str] = None) -> str:
    
    print(f"\n🔍 Analyzing query: {user_message[:120]}...\n")

    previous_state_obj: Optional[GovernanceSessionState] = None
    if previous_state_json:
        try:
            previous_state_obj = GovernanceSessionState.model_validate_json(previous_state_json)
        except Exception:
            previous_state_obj = None

    # 1) Parse intent + actions
    parsed = parse_intent_and_actions(user_message, previous_state_obj)

    # 2) Map to steps
    mapped = map_parsed_to_steps(user_message, parsed, previous_state_obj)

    # 3) Workflow reasoning
    state = compute_workflow_state(mapped)

    step_details = state.step_details

    def _step_label(s_id: str) -> str:
        if not s_id:
            return ""
        rec = step_details.get(s_id)
        if not rec:
            return s_id
        return f"{s_id} ({rec.get('name', s_id)})"

    # Determine automation options: focus step + first next step that are automatable
    automation_options = []
    candidate_ids: List[str] = []
    if state.focus_step_id:
        candidate_ids.append(state.focus_step_id)
    candidate_ids.extend(state.next_step_ids)

    seen_auto = set()
    for sid in candidate_ids:
        if sid in seen_auto:
            continue
        seen_auto.add(sid)
        rec = step_details.get(sid)
        if rec and rec.get("automatable") and rec.get("automation_step"):
            automation_options.append(
                {
                    "step_id": sid,
                    "label": _step_label(sid),
                    "automation_step": rec["automation_step"],
                }
            )

    context = {
        "user_message": user_message,
        "intent": state.intent,
        "is_starting_question": state.is_starting_question,
        "needs_disambiguation": state.needs_disambiguation,
        "completed_steps": [
            {"step_id": s_id, "label": _step_label(s_id)}
            for s_id in state.completed_ids
        ],
        "in_progress_steps": [
            {"step_id": s_id, "label": _step_label(s_id)}
            for s_id in state.in_progress_ids
        ],
        "referenced_steps": [
            {"step_id": s_id, "label": _step_label(s_id)}
            for s_id in state.referenced_ids
        ],
        "anchor_step": {
            "step_id": state.anchor_step_id,
            "label": _step_label(state.anchor_step_id),
        }
        if state.anchor_step_id
        else None,
        "focus_step": {
            "step_id": state.focus_step_id,
            "label": _step_label(state.focus_step_id),
        }
        if state.focus_step_id
        else None,
        "next_steps": [
            {"step_id": s_id, "label": _step_label(s_id)}
            for s_id in state.next_step_ids
        ],
        "missing_prerequisites": [
            {"step_id": s_id, "label": _step_label(s_id)}
            for s_id in state.missing_prereq_ids
        ],
        "step_details": state.step_details,
        "prereq_edges": state.prereq_edges,
        "semantic_candidates": [
            {
                "step_id": c.step_id,
                "label": _step_label(c.step_id),
                "score": c.score,
            }
            for c in state.semantic_candidates
        ],
        # NEW: automation options
        "automation_options": automation_options,
    }

    response_prompt = f"""
You are a governance workflow assistant helping users navigate a structured set of steps.

The context you must use is:
{json.dumps(context, indent=2)}

DEFINITIONS:
- intent:
  - "what_next"       → user asks what to do next / what comes after.
  - "help_current"    → user wants help completing something they're doing now.
  - "ask_about_step"  → user wants to understand a specific step or process.
  - "what_missed"     → user asks which governance steps they might not have done yet.
- is_starting_question:
  - true if they are asking where/how to start the whole process from scratch.
- needs_disambiguation:
  - true if the question could genuinely map to more than one governance step.
- focus_step:
  - the main step you should explain in detail (if not null).
- next_steps:
  - the step(s) the user should do next according to the workflow graph or canonical order.
- missing_prerequisites:
  - Steps that are MANDATORY in the workflow graph
  - AND are NOT mentioned as completed or in-progress by the user.
  - IMPORTANT: you DO NOT know whether they are actually done; you only know they have not been mentioned.
- semantic_candidates:
  - Top few steps that semantically match the user's question, each with step_id, label, and a similarity score.
- automation_options:
  - A (possibly empty) list of steps for which an automation helper is available.
  - Each item has: step_id, label, automation_step (e.g. "validate_name").

HARD RULES:

0) If needs_disambiguation = true:
   - This means the question could genuinely map to more than one governance step.
   - In this case, you MUST NOT pretend to know the single correct step.
   - DO NOT say "Your next step is ..." or act as if one specific step is definitely correct.
   - INSTEAD:
     * Briefly acknowledge the ambiguity, e.g. "This could relate to a couple of different governance steps."
     * List 2–3 candidate steps from semantic_candidates, e.g.:
         - "<step_id>: <step_name> — <one-sentence paraphrase of purpose>"
     * Ask 1–2 short clarifying questions to help the user choose which one applies.
       For example:
         - "Does your question relate more to registering the product so others can discover it,
            or to aggregating evidence for a final go/no-go decision?"
     * Keep it concise and practical.
   - After asking for clarification, do NOT attempt to fully explain a single step yet.
   - The follow-up user message will then be processed as a new query.

1) When intent = "ask_about_step" AND needs_disambiguation = false:
   - You MUST focus your explanation on focus_step (if not null).
   - Clearly state that step's ID and name, e.g. "For S21: <step name>...".
   - Use that step's purpose and description from step_details to explain 2–4 sentences
     of what they need to do / what it involves.
   - You MAY mention prerequisites and next steps, but as context or checks, not as the main focus.

2) When intent = "help_current" AND needs_disambiguation = false:
   - If focus_step is not null, your answer MUST describe how to complete focus_step.
   - Use the Nodes 'description' as your main source of detail (summarise and structure it).
   - You MAY mention what typically comes next at the end.

3) When intent = "what_next" AND needs_disambiguation = false:
   - If next_steps is not empty, your primary "next step" MUST be the first item of next_steps.
   - You MUST explicitly phrase it as: "Your next step is <step_id>: <step_name>."
   - Describe that next step using its description and purpose from step_details.
   - You MUST NOT describe any step in missing_prerequisites as the user's "next step".
   - You MUST NOT say or imply that a step in missing_prerequisites is definitely missing or not done.
   - You may still mention missing prerequisites as checks.

   - If is_starting_question = true and there are no completed steps,
     it is reasonable to suggest the very first governance step as their starting point.

4) When intent = "what_missed":
   - Do NOT assert that anything is definitely missing.
   - Instead, list candidate prerequisite steps from missing_prerequisites as things to double-check.
   - Use wording like:
        "Based on the steps you mentioned, here are some governance steps to double-check..."
   - For each, show step_id + name and brief purpose.

5) Missing prerequisites:
   - Treat missing_prerequisites as steps the user has NOT MENTIONED, not as steps they definitely skipped.
   - NEVER write sentences like:
       * "It looks like S1 is still missing."
       * "You haven't completed S1."
       * "You need to complete S1 first."
   - INSTEAD, only ask gentle confirmation questions and speak conditionally. For example:
       * "As a quick check, if you haven't already done S1, it's typically a prerequisite for this step."
   - You may adapt or reuse probe questions from prereq_edges when asking.

6) If there are no completed or in-progress steps, intent = "what_next",
   and is_starting_question = true AND needs_disambiguation = false:
   - Suggest the first step in the workflow as their starting point.

7) Automation (IMPORTANT):
   - You MUST NOT run any automation yourself in this response.
   - Instead, if automation_options is NON-EMPTY, you should:
       * Briefly mention that there is an automation helper available for that step, using the automation_step name.
       * Explain, in plain language, what that helper can do (e.g. "check your data-product name against the standard").
       * Invite the user to ask you to run it, e.g.:
           "If you'd like, I can run an automated naming check for your DOI — just say something like
            'run the naming check' or 'please validate the name'."
   - Do NOT assume the user wants automation; always ask.

8) Tone and structure:
   - Start with a brief, friendly acknowledgement of their question/progress.
   - Then clearly:
       * either explain the focus_step, or
       * explain the next step, or
       * ask for clarification between candidate steps (when needs_disambiguation = true).
   - Use bullet points if it helps readability.
   - Keep it concise but practical (no fluff).

Now, write your response to the user.
    """

    final_resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful, practical data governance assistant."},
            {"role": "user", "content": response_prompt},
        ],
        temperature=0,
        max_tokens=900,
    )

    answer = final_resp.choices[0].message.content

    # Determine last automatable step for memory (first in automation_options if any)
    auto_step_id = automation_options[0]["step_id"] if automation_options else None
    auto_step_key = automation_options[0]["automation_step"] if automation_options else None

    updated_state = GovernanceSessionState(
        last_intent=state.intent,
        last_focus_step_id=state.focus_step_id,
        last_anchor_step_id=state.anchor_step_id,
        last_next_step_ids=state.next_step_ids,
        last_completed_ids=state.completed_ids,
        last_in_progress_ids=state.in_progress_ids,
        last_referenced_ids=state.referenced_ids,
        last_automatable_step_id=auto_step_id,
        last_automation_step=auto_step_key,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )

    tool_output = GovernanceToolResult(
        answer=answer,
        updated_state=updated_state,
    )

    return tool_output.model_dump_json()

# Execute an automation tool from the registry.
@function_tool
def execute_automation(
    automation_step: str,
    user_input: Optional[str] = None
) -> str:

    # Get automation info
    info = get_automation_info(automation_step)
    if not info:
        available = list(list_available_automations().keys())
        return json.dumps({
            "status": "error",
            "formatted_message": f"❌ Unknown automation: {automation_step}. Available: {', '.join(available)}",
            "raw_result": {"error": "unknown_automation", "available": available}
        })
    
    # Execute automation
    result = run_automation_step(automation_step, user_input)
    
    # Format for user
    formatted = format_automation_result_for_user(result)
    
    return json.dumps({
        "status": result.get("status"),
        "formatted_message": formatted,
        "raw_result": result
    })


########## Governance Agent #########

governance_agent = Agent(
    name="Governance Assistant",
    instructions="""You are a governance workflow assistant.

For EVERY user message:
1. Call analyze_user_query(user_message, previous_state_json) 
2. Return the GovernanceToolResult it provides

That's it. Just call the tool and return its result.
The system will handle automation detection separately.""",
    tools=[analyze_user_query],  # Only analyze tool - automation handled elsewhere
    model=shared_model,
    output_type=GovernanceToolResult,
)


########## User interface with session-based memory #########

# Detect if user is requesting automation using reliable Python logic
def detect_automation_request(message: str) -> bool:
    msg_lower = message.lower()
    
    # Check for action words
    action_words = ["run", "execute", "validate", "check", "perform"]
    has_action = any(word in msg_lower for word in action_words)
    
    # Check for data product name pattern (AL####.Name)
    has_name = bool(re.search(r'al\d+\.\w+', msg_lower))
    
    # Check for type keywords
    has_type = any(t in msg_lower for t in ["odp", "fdp", "cdp"])
    
    return has_action and has_name and has_type

# Extract automation parameters from message using Python regex
def extract_automation_params(message: str) -> Optional[str]:
    # Extract name 
    name_match = re.search(r'(AL\d+\.\w+)', message, re.IGNORECASE)
    if not name_match:
        return None
    
    name = name_match.group(1)
    
    # Extract type
    msg_lower = message.lower()
    if "odp" in msg_lower:
        type_val = "ODP"
    elif "fdp" in msg_lower:
        type_val = "FDP"
    elif "cdp" in msg_lower:
        type_val = "CDP"
    else:
        return None
    
    # Return as JSON string
    return json.dumps({"name": name, "type": type_val})

# Process governance query with Python-based automation detection.
async def process_governance_query(
    user_message: str,
    session_id: str = "default_session",
) -> str:
    
    print("\n" + "=" * 80)
    print(f" GOVERNANCE QUERY (session: {session_id})")
    print(f" {user_message}")
    print("=" * 80)

    previous_state = SESSION_STATE.get(session_id)
    previous_state_json = json.dumps(previous_state) if previous_state is not None else None

    user_payload = {
        "user_message": user_message,
        "previous_state_json": previous_state_json,
    }

    # Step 1: Get workflow analysis from agent
    result = await Runner.run(governance_agent, json.dumps(user_payload))

    # Extract result
    if isinstance(result.final_output, GovernanceToolResult):
        answer = result.final_output.answer
        updated_state = result.final_output.updated_state.model_dump()
        SESSION_STATE[session_id] = updated_state
        print("   ✓ Workflow analysis complete")
    elif isinstance(result.final_output, str):
        try:
            parsed = json.loads(result.final_output)
            answer = parsed.get("answer", result.final_output)
            updated_state = parsed.get("updated_state", {})
            if updated_state:
                SESSION_STATE[session_id] = updated_state
        except:
            answer = result.final_output
            updated_state = {}
            SESSION_STATE[session_id] = {}
    else:
        answer = str(result.final_output)
        updated_state = {}
        SESSION_STATE[session_id] = {}
    
    # Step 2: Check for automation request (Python, not LLM!)
    if detect_automation_request(user_message):
        print("   🤖 Automation request detected (Python detection)")
        
        # Get automation step from state
        automation_step = updated_state.get("last_automation_step")
        
        if automation_step:
            print(f"   → Automation available: {automation_step}")
            
            # Extract parameters using Python
            params_json = extract_automation_params(user_message)
            
            if params_json:
                print(f"   → Extracted params: {params_json}")
                
                # Execute automation directly using registry functions (not the tool wrapper!)
                try:
                    # Call the automation registry directly
                    result = run_automation_step(automation_step, params_json)
                    formatted_message = format_automation_result_for_user(result)
                    
                    # Replace answer with automation result
                    answer = formatted_message
                    print("   ✓ Automation executed successfully")
                except Exception as e:
                    print(f"   ⚠️ Automation execution error: {e}")
                    answer = f"There was an error executing the automation: {str(e)}"
            else:
                print("   ⚠️ Could not extract parameters from message")
        else:
            print("   ⚠️ No automation available for current step")
    
    print("=" * 80 + "\n")
    return answer


######## Local demo ########

async def main():
    print("\n" + "=" * 80)
    print(" GOVERNANCE Q&A PIPELINE WITH MEMORY + EMBEDDINGS + AUTOMATION")
    print("=" * 80)

    demos = [
        ("session_flow", "I'm working on the DOI form, can you remind me what to do?"),
        ("session_flow", "Yes, can you run the naming check for me?"),
        ("session_new", "I think I need to log something for this new data thing—where do I start?"),
    ]

    for i, (sess, query) in enumerate(demos, 1):
        print(f"\n{'=' * 80}")
        print(f"DEMO {i} (session: {sess})")
        print(f"{'=' * 80}\n")

        response = await process_governance_query(query, session_id=sess)
        print(f"\n ANSWER:\n{response}\n")

        if i < len(demos):
            print("\n Next demo...\n")
            await asyncio.sleep(1)

    print("\n" + "=" * 80)
    print(" DEMOS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())