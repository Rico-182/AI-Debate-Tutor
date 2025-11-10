# app/main.py
import os
from datetime import datetime
from typing import Literal, Dict, List, Optional
import json
from uuid import uuid4, UUID
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------- Types ----------
load_dotenv()
Speaker = Literal["user", "assistant"]
Status = Literal["active", "completed"]

app = FastAPI(title="Debate MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- In-memory "DB" ----------
class Debate(BaseModel):
    id: UUID
    title: Optional[str] = None
    num_rounds: int
    starter: Speaker
    current_round: int = 1
    next_speaker: Speaker
    status: Status = "active"
    created_at: datetime
    updated_at: datetime

class Message(BaseModel):
    id: UUID
    debate_id: UUID
    round_no: int
    speaker: Speaker
    content: str
    created_at: datetime

DEBATES: Dict[UUID, Debate] = {}
MESSAGES: Dict[UUID, List[Message]] = {}  # keyed by debate_id
SCORES: Dict[UUID, "ScoreBreakdown"] = {}

# ---------- Schemas (I/O) ----------
class DebateCreate(BaseModel):
    title: Optional[str] = None
    num_rounds: int = Field(ge=1, le=3)
    starter: Speaker

class DebateOut(BaseModel):
    id: UUID
    title: Optional[str]
    num_rounds: int
    starter: Speaker
    current_round: int
    next_speaker: Speaker
    status: Status
    created_at: datetime
    updated_at: datetime

class MessageOut(BaseModel):
    id: UUID
    round_no: int
    speaker: Speaker
    content: str
    created_at: datetime

class DebateWithMessages(DebateOut):
    messages: List[MessageOut]

class TurnIn(BaseModel):
    speaker: Speaker
    content: str = Field(min_length=1)

class TurnOut(BaseModel):
    round_no: int
    accepted: bool
    next_speaker: Optional[Speaker]
    current_round: int
    status: Status
    message_id: UUID

class TranscribeOut(BaseModel):
    text: str
    language: Optional[str] = None


class ScoreMetrics(BaseModel):
    clarity: float
    structure: float
    engagement: float
    balance: float


class ScoreBreakdown(BaseModel):
    overall: float
    metrics: ScoreMetrics
    feedback: str = "No overall feedback provided."
    clarity_feedback: str = "No clarity feedback provided."
    structure_feedback: str = "No structure feedback provided."
    engagement_feedback: str = "No engagement feedback provided."
    balance_feedback: str = "No balance feedback provided."


class ScoreOut(ScoreBreakdown):
    debate_id: UUID

# ---------- Helpers ----------
def second_speaker_for_round(starter: Speaker) -> Speaker:
    return "assistant" if starter == "user" else "user"

def _append_message_and_advance(debate: Debate, speaker: Speaker, content: str) -> Message:
    """Save message, advance debate state; returns the saved message."""
    mid = uuid4()
    msg = Message(
        id=mid,
        debate_id=debate.id,
        round_no=debate.current_round,
        speaker=speaker,
        content=content.strip(),
        created_at=datetime.utcnow(),
    )
    MESSAGES[debate.id].append(msg)

    # Advance state
    if speaker == second_speaker_for_round(debate.starter):
        if debate.current_round >= debate.num_rounds:
            debate.status = "completed"
        else:
            debate.current_round += 1
            debate.next_speaker = debate.starter
    else:
        debate.next_speaker = "assistant" if speaker == "user" else "user"

    debate.updated_at = datetime.utcnow()
    return msg

# --- Optional OpenAI client (used for /auto-turn now; Whisper soon) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI  # OpenAI Python SDK ≥ 1.0
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None

def generate_ai_turn_text(debate: Debate, messages: List[Message]) -> str:
    """
    Produce assistant text. If OPENAI_API_KEY is set, call GPT-4o-mini.
    Otherwise return a simple stub so the endpoint still works for dev.
    """
    # Build concise chat context
    topic_context = f"Debate topic: {debate.title}" if debate.title else "Debate topic: General debate"
    sys = (
        "You are an APDA-style debater. Be clear, structured, and concise. "
        "Signpost arguments. 1–2 short paragraphs for turns. "
        f"{topic_context}. Engage with the arguments presented and provide thoughtful counterarguments."
    )
    print(topic_context)
    sys = (f'''
    You are DebaterGPT, a skilled competitive debater trained in APDA, Public Forum, and WSDC styles.

    When given a motion, argue as if you were in a live debate round, using clear signposting, logical flow, and rhetorical polish.

    If the prompt says “You: For”, argue in favor of the motion.

    If it says “You: Against”, argue against the motion for the user to rebut.

    Debate Style Rules:

    Start with a brief roadmap (“First, I’ll define terms, then present two contentions…”).

    Use contentions, warrants, and impacts in a clear structure.

    Weigh arguments explicitly (“We outweigh on magnitude and probability…”).

    Maintain a confident, persuasive tone fit for tournament debate.

    Always respond in the form of a debate speech, not an essay or explanation.

    The given topic is {topic_context}.
    ''')
    convo = []
    for m in messages:
        role = "user" if m.speaker == "user" else "assistant"
        tag = f"[Round {m.round_no} · {m.speaker.upper()}]"
        convo.append({"role": role, "content": f"{tag} {m.content}"})

    prompt_now = f"(Current round: {debate.current_round} of {debate.num_rounds}. You speak as {debate.next_speaker}. Provide a thoughtful response.)"

    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": sys}] + convo + [
                    {"role": "user", "content": prompt_now}
                ],
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass  # fall back to stub

    # Stub output for local dev without API key
    return f"(AI {debate.next_speaker} R{debate.current_round}) Brief, signposted response."

def transcribe_with_whisper(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    if not client:
        raise RuntimeError("OpenAI client not configured")
    from io import BytesIO
    bio = BytesIO(audio_bytes)
    bio.name = filename
    resp = client.audio.transcriptions.create(model="whisper-1", file=bio)
    return resp.text


def compute_debate_score(debate: Debate, messages: List[Message]) -> ScoreBreakdown:
    if not client:
        raise HTTPException(503, "Scoring requires OpenAI API key")

    if not messages:
        return ScoreBreakdown(
            overall=0.0,
            metrics=ScoreMetrics(clarity=0.0, structure=0.0, engagement=0.0, balance=0.0),
            feedback="No debate content available to score.",
            clarity_feedback="No clarity feedback available.",
            structure_feedback="No structure feedback available.",
            engagement_feedback="No engagement feedback available.",
            balance_feedback="No balance feedback available."
        )

    convo: List[Dict[str, str]] = []
    for msg in messages:
        role = "user" if msg.speaker == "user" else "assistant"
        speaker = msg.speaker.upper()
        convo.append({
            "role": role,
            "content": f"[Round {msg.round_no} · {speaker}] {msg.content}"
        })

    system_prompt = (
        "You are DebateJudgeGPT, an expert debate adjudicator across APDA, Public Forum, and WSDC formats.\n"
        "You will be given the full transcript of a debate between a human debater (`user`) and an AI sparring partner (`assistant`).\n"
        "Your task is to evaluate ONLY the human debater's performance. The AI serves purely as opposition context.\n\n"
        "Score the human on these metrics (0-100 each):\n"
        "1. Clarity & Delivery – vocal clarity, persuasive tone, pacing, accessibility of arguments.\n"
        "2. Structure & Organization – use of roadmaps, contention structure, logical flow, internal signposting.\n"
        "3. Engagement & Clash – responsiveness to the AI's points, refutation quality, impact weighing, comparative analysis.\n"
        "4. Strategic Balance & Completion – effective use of allotted rounds, time management, closing strength, and overall debate completeness.\n\n"
        "Provide:\n"
        "- `overall_score`: holistic score (0-100) for the human debater.\n"
        "- `feedback`: a comprehensive 4-6 sentence paragraph synthesizing strengths, critical weaknesses, and concrete next steps.\n"
        "- For each metric, include a numeric subscore (`*_score`) AND a two-sentence targeted coaching tip (`*_feedback`) with actionable guidance referencing specific debate behaviors.\n"
        "Return ONLY a JSON object with keys: overall_score, feedback, clarity_score, clarity_feedback, "
        "structure_score, structure_feedback, engagement_score, engagement_feedback, "
        "balance_score, balance_feedback.\n"
        "Do not include any additional text outside the JSON."
    )

    prompt = [
        {"role": "system", "content": system_prompt},
        *convo,
        {
            "role": "user",
            "content": (
                "Judge this debate according to the instructions. "
                "Be fair, constructive, and reference specific debate behaviors in your feedback."
            ),
        },
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            temperature=0.3,
        )
    except Exception as exc:
        raise HTTPException(502, f"Scoring failed: {exc}") from exc

    raw = resp.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(502, f"Scoring response malformed: {raw}")

    def _get_float(key: str) -> float:
        value = parsed.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    metrics = ScoreMetrics(
        clarity=round(_get_float("clarity_score"), 1),
        structure=round(_get_float("structure_score"), 1),
        engagement=round(_get_float("engagement_score"), 1),
        balance=round(_get_float("balance_score"), 1),
    )

    overall = round(_get_float("overall_score"), 1)

    return ScoreBreakdown(
        overall=overall,
        metrics=metrics,
        feedback=parsed.get("feedback", "No overall feedback provided."),
        clarity_feedback=parsed.get("clarity_feedback", "No clarity feedback provided."),
        structure_feedback=parsed.get("structure_feedback", "No structure feedback provided."),
        engagement_feedback=parsed.get("engagement_feedback", "No engagement feedback provided."),
        balance_feedback=parsed.get("balance_feedback", "No balance feedback provided."),
    )


def _score_out_from_breakdown(debate_id: UUID, breakdown: ScoreBreakdown) -> ScoreOut:
    # Ensure we have a ScoreBreakdown instance with all fields populated
    if not isinstance(breakdown, ScoreBreakdown):
        breakdown = ScoreBreakdown.model_validate(breakdown)

    metrics = breakdown.metrics or ScoreMetrics(
        clarity=0.0, structure=0.0, engagement=0.0, balance=0.0
    )

    return ScoreOut(
        debate_id=debate_id,
        overall=breakdown.overall,
        metrics=metrics,
        feedback=breakdown.feedback,
        clarity_feedback=breakdown.clarity_feedback,
        structure_feedback=breakdown.structure_feedback,
        engagement_feedback=breakdown.engagement_feedback,
        balance_feedback=breakdown.balance_feedback,
    )

# ---------- 1) Create debate ----------
@app.post("/v1/debates", response_model=DebateOut, status_code=201)
def create_debate(body: DebateCreate):
    did = uuid4()
    now = datetime.utcnow()
    debate = Debate(
        id=did,
        title=body.title,
        num_rounds=body.num_rounds,
        starter=body.starter,
        current_round=1,
        next_speaker=body.starter,
        status="active",
        created_at=now,
        updated_at=now,
    )
    DEBATES[did] = debate
    MESSAGES[did] = []
    return debate

# ---------- 2) Submit a turn ----------
@app.post("/v1/debates/{debate_id}/turns", response_model=TurnOut)
def submit_turn(debate_id: UUID, body: TurnIn):
    debate = DEBATES.get(debate_id)
    if not debate:
        raise HTTPException(404, "Debate not found")
    if debate.status != "active":
        raise HTTPException(400, f"Debate is {debate.status}")
    if body.speaker != debate.next_speaker:
        raise HTTPException(409, f"It is {debate.next_speaker}'s turn.")

    msg = _append_message_and_advance(debate, body.speaker, body.content)

    return TurnOut(
        round_no=msg.round_no,
        accepted=True,
        next_speaker=debate.next_speaker if debate.status == "active" else None,
        current_round=debate.current_round,
        status=debate.status,
        message_id=msg.id,
    )

# ---------- 3) GET state + messages ----------
@app.get("/v1/debates/{debate_id}", response_model=DebateWithMessages)
def get_debate(debate_id: UUID):
    debate = DEBATES.get(debate_id)
    if not debate:
        raise HTTPException(404, "Debate not found")
    msgs = [
        MessageOut(
            id=m.id,
            round_no=m.round_no,
            speaker=m.speaker,
            content=m.content,
            created_at=m.created_at,
        )
        for m in MESSAGES.get(debate_id, [])
    ]
    return DebateWithMessages(**debate.model_dump(), messages=msgs)

# ---------- 4) Auto-turn (assistant generates its move) ----------
class AutoTurnOut(BaseModel):
    message_id: UUID
    content: str
    round_no: int
    next_speaker: Optional[Speaker]
    current_round: int
    status: Status

@app.post("/v1/debates/{debate_id}/auto-turn", response_model=AutoTurnOut)
def auto_turn(debate_id: UUID):
    debate = DEBATES.get(debate_id)
    if not debate:
        raise HTTPException(404, "Debate not found")
    if debate.status != "active":
        raise HTTPException(400, f"Debate is {debate.status}")
    if debate.next_speaker != "assistant":
        raise HTTPException(409, "It's not the assistant's turn.")

    history = MESSAGES.get(debate_id, [])
    ai_text = generate_ai_turn_text(debate, history)

    msg = _append_message_and_advance(debate, "assistant", ai_text)

    return AutoTurnOut(
        message_id=msg.id,
        content=msg.content,
        round_no=msg.round_no,
        next_speaker=debate.next_speaker if debate.status == "active" else None,
        current_round=debate.current_round,
        status=debate.status,
    )

# ---------- 5) Finish early ----------
class FinishOut(BaseModel):
    status: Status
    current_round: int
    next_speaker: Optional[Speaker]

@app.post("/v1/debates/{debate_id}/finish", response_model=FinishOut)
def finish_debate(debate_id: UUID):
    debate = DEBATES.get(debate_id)
    if not debate:
        raise HTTPException(404, "Debate not found")
    debate.status = "completed"
    debate.updated_at = datetime.utcnow()
    # Optional: nullify next speaker when completed
    next_sp = None
    return FinishOut(status=debate.status, current_round=debate.current_round, next_speaker=next_sp)

@app.post("/v1/transcribe", response_model=TranscribeOut)
async def transcribe(file: UploadFile = File(...)):
    # Basic content-type check
    if not file.content_type or "audio" not in file.content_type:
        # Some browsers send octet-stream; still allow if filename looks like audio
        allowed = (file.filename or "").lower().endswith((".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"))
        if not allowed:
            raise HTTPException(400, "Please upload an audio file")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty file")

    try:
        text = transcribe_with_whisper(audio_bytes, filename=file.filename or "audio.wav")
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        # Surface a concise error; log details in real app
        raise HTTPException(502, f"Transcription failed: {e}")

    return TranscribeOut(text=text)


@app.post("/v1/debates/{debate_id}/score", response_model=ScoreOut)
def score_debate(debate_id: UUID):
    debate = DEBATES.get(debate_id)
    if not debate:
        raise HTTPException(404, "Debate not found")
    if debate.status != "completed":
        raise HTTPException(409, "Debate must be completed before scoring")

    messages = MESSAGES.get(debate_id, [])
    breakdown = compute_debate_score(debate, messages)
    SCORES[debate_id] = breakdown

    return _score_out_from_breakdown(debate_id, breakdown)


@app.get("/v1/debates/{debate_id}/score", response_model=ScoreOut)
def get_score(debate_id: UUID):
    debate = DEBATES.get(debate_id)
    if not debate:
        raise HTTPException(404, "Debate not found")

    breakdown = SCORES.get(debate_id)
    if not breakdown:
        raise HTTPException(404, "Score not found. Score the debate first.")

    # Backfill missing feedback fields for legacy scores
    if not getattr(breakdown, "feedback", None):
        messages = MESSAGES.get(debate_id, [])
        try:
            breakdown = compute_debate_score(debate, messages)
            SCORES[debate_id] = breakdown
        except HTTPException:
            breakdown = ScoreBreakdown(
                overall=getattr(breakdown, "overall", 0.0),
                metrics=getattr(
                    breakdown,
                    "metrics",
                    ScoreMetrics(clarity=0.0, structure=0.0, engagement=0.0, balance=0.0),
                ),
            )

    return _score_out_from_breakdown(debate_id, breakdown)


# ---------- Health ----------
@app.get("/v1/health")
def health():
    return {"status": "ok"}
