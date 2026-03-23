import os
import json
from typing import Dict, List

import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

load_dotenv()

app = FastAPI(title="SafeBuild Agent App")
templates = Jinja2Templates(directory="templates")

# Environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")
model_deployment_name = os.getenv("MODEL_DEPLOYMENT_NAME")

azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

# DAB REST base URL
dab_base_url = os.getenv("DAB_BASE_URL")

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_base_url,
)

# Simple in-memory conversation store
# Note: resets on container restart/redeploy
conversation_store: Dict[str, List[dict]] = {}
MAX_HISTORY_MESSAGES = 12


class InspectionCreate(BaseModel):
    site: str
    inspection_datetime: str
    inspector: str
    hazard: str
    severity: str
    description: str
    recommended_action: str
    status: str


class ChatRequest(BaseModel):
    conversation_id: str
    message: str


class ResetChatRequest(BaseModel):
    conversation_id: str


def get_conversation_history(conversation_id: str) -> List[dict]:
    return conversation_store.get(conversation_id, [])


def append_to_conversation(conversation_id: str, role: str, content: str) -> None:
    if conversation_id not in conversation_store:
        conversation_store[conversation_id] = []

    conversation_store[conversation_id].append({
        "role": role,
        "content": content
    })

    if len(conversation_store[conversation_id]) > MAX_HISTORY_MESSAGES:
        conversation_store[conversation_id] = conversation_store[conversation_id][-MAX_HISTORY_MESSAGES:]


def reset_conversation(conversation_id: str) -> None:
    conversation_store[conversation_id] = []


def clean_json_text(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def search_knowledge_base(query: str, top: int = 3) -> str:
    if not azure_search_endpoint or not azure_search_admin_key or not azure_search_index_name:
        return ""

    url = (
        f"{azure_search_endpoint}/indexes/"
        f"{azure_search_index_name}/docs/search?api-version=2025-09-01"
    )

    headers = {
        "Content-Type": "application/json",
        "api-key": azure_search_admin_key,
    }

    payload = {
        "search": query,
        "top": top,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    chunks = []
    for doc in data.get("value", []):
        chunks.append(str(doc))

    return "\n\n".join(chunks[:top])


def build_dab_api_url() -> str:
    if not dab_base_url:
        raise ValueError("DAB_BASE_URL not configured")

    base = dab_base_url.strip().rstrip("/")

    if not base.startswith("http://") and not base.startswith("https://"):
        base = f"https://{base}"

    return f"{base}/api/SafetyInspections"


def inspection_to_dict(inspection: InspectionCreate) -> dict:
    if hasattr(inspection, "model_dump"):
        return inspection.model_dump()
    return inspection.dict()


def create_inspection_in_db(payload: dict) -> dict:
    url = build_dab_api_url()

    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    response.raise_for_status()

    return response.json()


def build_history_text(history: List[dict]) -> str:
    if not history:
        return "No prior conversation."

    lines = []
    for item in history[-8:]:
        role = item.get("role", "user")
        content = item.get("content", "")
        lines.append(f"{role.upper()}: {content}")

    return "\n".join(lines)


def detect_intent(user_text: str, history: List[dict]) -> str:
    keyword_hits = [
        "report hazard",
        "report a hazard",
        "log inspection",
        "log an inspection",
        "create inspection",
        "create an inspection",
        "record inspection",
        "submit inspection",
        "new inspection",
        "unsafe condition",
        "hazard observed",
        "inspection log",
        "log a safety issue",
        "create a safety inspection",
        "record a hazard",
        "add inspection",
        "submit a hazard",
        "file an inspection",
    ]

    text_lower = user_text.lower()
    if any(keyword in text_lower for keyword in keyword_hits):
        return "create_inspection"

    history_text = build_history_text(history)

    system_message = (
        "Classify the user's latest request into one of two labels only: "
        "create_inspection or rag_answer. "
        "Return only one label and nothing else. "
        "Use create_inspection when the user is asking to log, record, create, submit, "
        "or report a safety inspection or hazard record for storage. "
        "Also use create_inspection when the latest user message is a follow-up that adds or changes "
        "inspection details within an existing inspection-reporting conversation. "
        "Use rag_answer when the user is asking a question, seeking guidance, explanation, "
        "PPE advice, procedures, hazards education, or general safety information."
    )

    user_message = f"""
Conversation history:
{history_text}

Latest user message:
{user_text}
"""

    response = client.chat.completions.create(
        model=model_deployment_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )

    label = (response.choices[0].message.content or "").strip().lower()

    if "create_inspection" in label:
        return "create_inspection"

    return "rag_answer"


def extract_inspection_from_text(user_text: str, history: List[dict]) -> dict:
    history_text = build_history_text(history)

    system_message = (
        "You extract structured construction safety inspection records from user input. "
        "Return only valid JSON with exactly these keys: "
        "site, inspection_datetime, inspector, hazard, severity, description, "
        "recommended_action, status. "
        "Use the conversation history when the latest message is a follow-up. "
        "Rules: "
        "1. Output JSON only. "
        "2. Do not include markdown fences. "
        "3. If the user does not provide a value, infer a reasonable business-safe placeholder. "
        "4. inspection_datetime must be ISO format like 2026-03-19T10:00:00. "
        "5. status should usually be Open unless the user states otherwise. "
        "6. severity should be one of Low, Medium, High, Critical."
    )

    user_message = f"""
Conversation history:
{history_text}

Latest user input:
{user_text}
"""

    response = client.chat.completions.create(
        model=model_deployment_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )

    content = response.choices[0].message.content
    cleaned = clean_json_text(content)
    parsed = json.loads(cleaned)

    required_fields = [
        "site",
        "inspection_datetime",
        "inspector",
        "hazard",
        "severity",
        "description",
        "recommended_action",
        "status",
    ]

    for field in required_fields:
        if field not in parsed or parsed[field] is None or str(parsed[field]).strip() == "":
            raise ValueError(f"Missing required extracted field: {field}")

    return parsed


def answer_safety_question(question: str, history: List[dict]) -> str:
    kb_context = search_knowledge_base(question)
    history_text = build_history_text(history)

    system_message = (
        "You are SafeBuild Safety Agent, a construction safety assistant "
        "for a construction company. Answer questions about hazards, PPE, "
        "inspections, and safety procedures. Be practical, concise, and structured. "
        "Do not invent regulations or citations. When appropriate, include both risks "
        "and prevention steps. Use the retrieved knowledge base context when relevant. "
        "Use conversation history when the latest question depends on prior context. "
        "If the answer is not contained in the retrieved context, say you are not sure."
    )

    user_message = f"""
Conversation history:
{history_text}

Retrieved context:
{kb_context}

Latest user question:
{question}
"""

    response = client.chat.completions.create(
        model=model_deployment_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ask")
def ask_agent(question: str):
    try:
        if not question or not question.strip():
            return {"error": "Question is required"}

        intent = detect_intent(question, [])

        if intent == "create_inspection":
            extracted_payload = extract_inspection_from_text(question, [])
            insert_result = create_inspection_in_db(extracted_payload)

            created_row = None
            response_value = insert_result.get("value", [])
            if response_value and isinstance(response_value, list):
                created_row = response_value[0]

            return {
                "action": "create_inspection",
                "question": question,
                "message": "Inspection record created successfully. Thank you for submitting the safety inspection.",
                "inspection_id": created_row.get("id") if created_row else None,
                "submitted_payload": extracted_payload,
            }

        answer = answer_safety_question(question, [])

        return {
            "action": "rag_answer",
            "question": question,
            "answer": answer,
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        conversation_id = request.conversation_id.strip()
        message = request.message.strip()

        if not conversation_id:
            return {"error": "conversation_id is required"}

        if not message:
            return {"error": "message is required"}

        history = get_conversation_history(conversation_id)
        intent = detect_intent(message, history)

        append_to_conversation(conversation_id, "user", message)

        if intent == "create_inspection":
            updated_history = get_conversation_history(conversation_id)
            extracted_payload = extract_inspection_from_text(message, updated_history)
            insert_result = create_inspection_in_db(extracted_payload)

            created_row = None
            response_value = insert_result.get("value", [])
            if response_value and isinstance(response_value, list):
                created_row = response_value[0]

            assistant_message = "Inspection record created successfully. Thank you for submitting the safety inspection."
            if created_row and created_row.get("id") is not None:
                assistant_message += f" Inspection ID: {created_row.get('id')}."

            append_to_conversation(conversation_id, "assistant", assistant_message)

            return {
                "action": "create_inspection",
                "message": assistant_message,
                "inspection_id": created_row.get("id") if created_row else None,
                "submitted_payload": extracted_payload,
                "conversation_id": conversation_id,
            }

        updated_history = get_conversation_history(conversation_id)
        answer = answer_safety_question(message, updated_history)
        append_to_conversation(conversation_id, "assistant", answer)

        return {
            "action": "rag_answer",
            "message": answer,
            "conversation_id": conversation_id,
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/reset-chat")
def reset_chat(request: ResetChatRequest):
    try:
        conversation_id = request.conversation_id.strip()

        if not conversation_id:
            return {"error": "conversation_id is required"}

        reset_conversation(conversation_id)

        return {
            "message": "Conversation reset successfully.",
            "conversation_id": conversation_id,
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/create-inspection")
def create_inspection(inspection: InspectionCreate):
    try:
        payload = inspection_to_dict(inspection)
        result = create_inspection_in_db(payload)

        created_row = None
        response_value = result.get("value", [])
        if response_value and isinstance(response_value, list):
            created_row = response_value[0]

        return {
            "message": "Inspection record created successfully. Thank you for submitting the safety inspection.",
            "inspection_id": created_row.get("id") if created_row else None,
            "submitted_payload": payload,
        }

    except Exception as e:
        return {"error": str(e)}