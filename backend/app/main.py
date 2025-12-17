# app/main.py
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

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

# Import RAG functionality
from app.response import SimpleRAG, generate_debate_with_coach_loop, generate_rebuttal_speech

# ---------- Types ----------
load_dotenv()
Speaker = Literal["user", "assistant"]
Status = Literal["active", "completed"]

app = FastAPI(title="Debate MVP")

# CORS configuration - allow specific origins in production
CORS_ORIGINS_ENV = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = ["*"] if CORS_ORIGINS_ENV == "*" else [origin.strip() for origin in CORS_ORIGINS_ENV.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- RAG System ----------
# Global RAG instance - will be initialized on startup
rag_system: Optional[SimpleRAG] = None

@app.on_event("startup")
def startup_event():
    """Initialize RAG system with corpus on app startup"""
    global rag_system
    corpus_dir = os.getenv("SPEECH_CORPUS_DIR", "./app/corpus")

    print(f"[RAG] Initializing RAG system with corpus from: {corpus_dir}")
    rag_system = SimpleRAG()

    if os.path.exists(corpus_dir):
        rag_system.add_corpus_folder(corpus_dir, pattern=r".*\.txt$")
        print(f"[RAG] Indexed {len(rag_system.docs)} documents")
    else:
        print(f"[RAG] Warning: Corpus directory not found: {corpus_dir}")
        print(f"[RAG] RAG system initialized but no documents loaded")


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

# ---------- RAG Schemas ----------
class RAGSpeechRequest(BaseModel):
    motion: str = Field(min_length=1)
    side: Literal["Government", "Opposition"] = "Government"
    format: str = "WSDC"
    use_rag: bool = True
    top_k: int = Field(default=6, ge=1, le=20)
    min_score: float = Field(default=0.1, ge=0.0, le=1.0)
    model: str = "gpt-4o-mini"
    temp_low: float = Field(default=0.3, ge=0.0, le=2.0)
    temp_high: float = Field(default=0.8, ge=0.0, le=2.0)

class RAGRebuttalRequest(BaseModel):
    motion: str = Field(min_length=1)
    opponent_speech: str = Field(min_length=1)
    side: Literal["Government", "Opposition"] = "Opposition"
    format: str = "WSDC"
    use_rag: bool = True
    top_k: int = Field(default=6, ge=1, le=20)
    min_score: float = Field(default=0.1, ge=0.0, le=1.0)
    model: str = "gpt-4o-mini"
    temp_low: float = Field(default=0.3, ge=0.0, le=2.0)
    temp_high: float = Field(default=0.8, ge=0.0, le=2.0)

class RAGSpeechResponse(BaseModel):
    speech: str
    context_count: int
    avg_score: Optional[float] = None

class CorpusStatsResponse(BaseModel):
    total_documents: int
    corpus_available: bool

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
    Produce assistant text using RAG-powered generation when available.
    Falls back to basic GPT-4o-mini if RAG is not initialized.
    """
    motion = debate.title if debate.title else "General debate topic"

    # If we have RAG and this is a rebuttal (not the first turn)
    if rag_system and len(messages) > 0:
        # Determine side - assistant is Opposition if starter is user, Government if starter is assistant
        side = "Opposition" if debate.starter == "user" else "Government"

        # Get the last opponent message to rebut
        opponent_messages = [m for m in messages if m.speaker != "assistant"]
        if opponent_messages:
            last_opponent = opponent_messages[-1]
            try:
                result = generate_rebuttal_speech(
                    rag=rag_system,
                    motion=motion,
                    opponent_speech=last_opponent.content,
                    side=side,
                    format="WSDC",
                    use_rag=True,
                    top_k=6,
                    min_score=0.1,
                    model="gpt-4o-mini",
                    temp_low=0.3,
                    temp_high=0.8
                )
                return result["rebuttal_speech"]
            except Exception as e:
                print(f"[RAG] Rebuttal generation failed: {e}, falling back to basic generation")

    # If this is the first turn and we have RAG
    if rag_system and len(messages) == 0:
        side = "Government" if debate.starter == "assistant" else "Opposition"
        try:
            result = generate_debate_with_coach_loop(
                rag=rag_system,
                motion=motion,
                side=side,
                format="WSDC",
                use_rag=True,
                top_k=6,
                min_score=0.1,
                model="gpt-4o-mini",
                temperature_gen=0.5,
                temperature_rev=0.2,
                temp_low=0.3,
                temp_high=0.8
            )
            return result["initial_speech"]
        except Exception as e:
            print(f"[RAG] Speech generation failed: {e}, falling back to basic generation")

    # Fallback: use similar style to fine-tuned prompts in response.py
    topic_context = f"Debate topic: {motion}"
    sys = """You are a world-class competitive debater. Your goal is to WIN through sharp, incisive argumentation—not through aggression. Be strategic, precise, and respectful. Fill gaps with compelling real-world examples that any educated voter would recognize—think NYT front page, not academic journals. Use concrete mechanisms and numbers. Make clear, fair comparisons that demonstrate why your case is stronger. Sound like a human champion debater who wins through superior logic and analysis, not an essay or a robot.
If the
When delivering your speech:
- SIGNPOST HEAVILY: Label everything clearly
- Provide 2-3 well-developed contentions/arguments
- Each argument needs: PREMISE → 2-3 WELL-DEVELOPED MECHANISMS → IMPACT
- Develop EACH mechanism with 2-3 sentences (explain the causal chain, then elaborate)
- Weigh arguments explicitly on magnitude, probability, timeframe
- Zero filler, no pleasantries
- Sound human, not robotic - vary sentence length
- Use concrete examples voters recognize (Amazon, climate disasters, iPhone)
- NEVER cite sources - use examples as common knowledge
- Make it sound like you're SPEAKING, not reading an essay
- There is no point in saying stuff that both sides would agree to : for example when debating about aid, both teams can agree that "aid is a powerful tool that can either stabilize or destabilize regions". THis is useless.
"""
    convo = []
    for m in messages:
        role = "user" if m.speaker == "user" else "assistant"
        tag = f"[Round {m.round_no} · {m.speaker.upper()}]"
        convo.append({"role": role, "content": f"{tag} {m.content}"})

    # Determine if this is first speech or rebuttal
    opponent_messages = [m for m in messages if m.speaker != debate.next_speaker]
    if opponent_messages:
        # Rebuttal situation
        prompt_now = f"""Motion: {topic_context}
Current round: {debate.current_round} of {debate.num_rounds}

Deliver a rebuttal speech that:
1. Tears down the opponent's key arguments (address their strongest points first).
- When rebutting, think about NEGATING first. If they claim something, explain a proper reason why that something is NOT true
- If that's hard, mitigate it. If they claim sometthing, explain why it's not as big of an issue as they claimed. 
- If that's also too hard, last resort: concede to it, but explain why your argument is still more important. 
2. Presents 1 new constructive arguments, explicitly label this as an "extension" or "spike"
3. Does comparative weighing

Be sharp, precise, and demonstrate why your case is stronger."""
    else:
        # Opening speech
        prompt_now = f"""Motion: {topic_context}
Current round: {debate.current_round} of {debate.num_rounds}

Deliver an opening speech following the structure:
1. Opening hook (2-3 sentences with real-world example)
2. Framing & burdens
3. 2-3 contentions (each with premise, mechanisms, weighing)
4. Conclusion

Make your case compelling and well-structured."""

    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": sys}] + convo + [
                    {"role": "user", "content": prompt_now}
                ],
                temperature=0.7,  # Higher temp since no RAG context (matches response.py adaptive logic)
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
