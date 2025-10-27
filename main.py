import os
import json
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import db, create_document, get_documents
from schemas import Qvalue, Question, Interaction, Userprofile

# Optional Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
try:
    from groq import Groq  # type: ignore
    groq_client: Optional[Any] = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception:
    groq_client = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- RL helpers ----------
ACTIONS = ["upgrade", "downgrade", "maintain", "focus_weak", "focus_strong"]
DIFF_ORDER = ["easy", "medium", "hard"]

def clamp(num: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, num))

def bucket(v: float, low: float, high: float) -> str:
    if v < low:
        return "low"
    if v >= high:
        return "high"
    return "mid"


def profile_to_state(profile: Dict[str, Any]) -> str:
    pm = profile.get("performanceMetrics", {}) or {}
    accuracy = float(pm.get("accuracy", 0))
    speed = float(pm.get("speed", 0))
    consistency = float(pm.get("consistency", 0))
    return f"{bucket(accuracy,50,80)}_acc_{bucket(speed,40,70)}_spd_{bucket(consistency,50,75)}_con"


def choose_action_db(user_id: str, state: str, profile: Dict[str, Any], eps_base: float = 0.25) -> Dict[str, str]:
    # epsilon decay by consistency
    pm = profile.get("performanceMetrics", {}) or {}
    consistency = float(pm.get("consistency", 0))
    eps = clamp(eps_base - (consistency / 100.0) * (eps_base - 0.05), 0.05, eps_base)

    # load Q-values for this user and state
    qdocs = db["qvalue"].find({"user_id": user_id, "state": state})
    qmap: Dict[str, float] = {d.get("action"): float(d.get("q", 0.0)) for d in qdocs}

    # exploration vs exploitation
    import random
    if random.random() < eps or not qmap:
        action = random.choice(ACTIONS)
    else:
        action = max(ACTIONS, key=lambda a: qmap.get(a, 0.0))

    # difficulty transition
    curr = (profile.get("currentDifficulty") or "medium").lower()
    if curr not in DIFF_ORDER:
        curr = "medium"
    idx = DIFF_ORDER.index(curr)
    if action == "upgrade":
        idx = clamp(idx + 1, 0, len(DIFF_ORDER) - 1)
    elif action == "downgrade":
        idx = clamp(idx - 1, 0, len(DIFF_ORDER) - 1)
    difficulty = DIFF_ORDER[int(idx)]

    # choose concept focus
    topic = (profile.get("topic") or "arrays").lower()
    concept_mastery = (pm.get("conceptMastery") or {})
    weak = [k for k, v in concept_mastery.items() if float(v) < 70]
    strong = [k for k, v in concept_mastery.items() if float(v) >= 70]
    focus = topic
    if action == "focus_weak" and weak:
        focus = weak[0]
    elif action == "focus_strong" and strong:
        focus = strong[0]

    return {"action": action, "difficulty": difficulty, "focusConcept": focus}


# ---------- Question generation ----------

def fallback_question(topic: str, action: str, difficulty: str, focus: str) -> Dict[str, Any]:
    # basic deterministic fallback similar to the frontend version
    def arrays_easy():
        return {
            "title": "Running Sum of Array",
            "description": (
                "Given an array of integers, return the running sum (prefix sums).\n\n"
                "Constraints:\n- 1 <= n <= 1e5\n- O(n) time, O(1) extra space\n\n"
                "Input: space-separated integers\nOutput: space-separated running sums"
            ),
            "starterCode": (
                "def running_sum(nums):\n    # Write your code here\n    pass\n\n"
                "if __name__ == '__main__':\n    import sys\n    nums = list(map(int, sys.stdin.read().strip().split()))\n    res = running_sum(nums)\n    print(' '.join(map(str, res)))\n"
            ),
            "solutionCode": (
                "def running_sum(nums):\n    s = 0\n    for i in range(len(nums)):\n        s += nums[i]\n        nums[i] = s\n    return nums\n\n"
                "if __name__ == '__main__':\n    import sys\n    nums = list(map(int, sys.stdin.read().strip().split()))\n    res = running_sum(nums)\n    print(' '.join(map(str, res)))\n"
            ),
            "expectedOutput": "1 3 6 10",
            "testCases": [
                {"input": "1 2 3 4", "expectedOutput": "1 3 6 10"},
                {"input": "5", "expectedOutput": "5"},
            ],
            "tags": ["arrays", "prefix-sum", difficulty, action],
        }

    def strings_easy():
        return {
            "title": "Valid Anagram",
            "description": (
                "Given two strings s and t, return true if t is an anagram of s. Assume lowercase ASCII.\n\n"
                "Constraints:\n- 1 <= |s|,|t| <= 1e5\n- O(n) time\n\nInput: two lines s and t\nOutput: true/false"
            ),
            "starterCode": (
                "def is_anagram(s, t):\n    # Write your code here\n    pass\n\n"
                "if __name__ == '__main__':\n    import sys\n    s, t = sys.stdin.read().strip().split('\n')\n    print(str(is_anagram(s, t)).lower())\n"
            ),
            "solutionCode": (
                "def is_anagram(s, t):\n    if len(s) != len(t):\n        return False\n    freq = [0]*26\n    for ch in s:\n        freq[ord(ch)-97] += 1\n    for ch in t:\n        i = ord(ch)-97\n        freq[i] -= 1\n        if freq[i] < 0:\n            return False\n    return True\n\n"
                "if __name__ == '__main__':\n    import sys\n    s, t = sys.stdin.read().strip().split('\n')\n    print(str(is_anagram(s, t)).lower())\n"
            ),
            "expectedOutput": "true",
            "testCases": [
                {"input": "anagram\nnagaram", "expectedOutput": "true"},
                {"input": "rat\ncar", "expectedOutput": "false"},
            ],
            "tags": ["strings", "hashing", difficulty, action],
        }

    topic_key = "strings" if ("string" in focus.lower()) else ("arrays")
    if topic_key == "arrays":
        base = arrays_easy()
    else:
        base = strings_easy()
    base["tags"].append(focus)
    return base


def generate_with_groq(topic: str, action: str, difficulty: str, focus: str) -> Dict[str, Any]:
    if not groq_client:
        raise RuntimeError("Groq not configured")
    sys_prompt = (
        "You are an expert Python problem setter. Given a target concept and difficulty, "
        "produce a JSON object with keys: title, description, starterCode, solutionCode, "
        "expectedOutput (optional), testCases (array of {input, expectedOutput}), tags (array). "
        "Keep description concise with constraints and IO format."
    )
    user_prompt = (
        f"Topic: {topic}\nTarget concept: {focus}\nDifficulty: {difficulty}\n"
        f"Policy hint: {action}. Return ONLY a JSON object."
    )
    completion = groq_client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        response_format={"type": "json_object"},
        max_tokens=1200,
    )
    content = completion.choices[0].message.content
    data = json.loads(content)
    # basic validation
    for key in ["title", "description", "starterCode", "solutionCode", "testCases", "tags"]:
        if key not in data:
            raise ValueError("Incomplete response from Groq")
    return data


# ---------- API models ----------
class GenerateRequest(BaseModel):
    user_id: str
    profile: Dict[str, Any]

class GenerateResponse(BaseModel):
    decision: Dict[str, Any]
    state: str
    question_id: str
    question: Dict[str, Any]

class SubmitRequest(BaseModel):
    user_id: str
    question_id: str
    state: str
    action: str
    # outcome
    correct: Optional[bool] = None
    time_ms: Optional[int] = None
    rating: Optional[int] = None
    next_profile: Optional[Dict[str, Any]] = None

class HistoryItem(BaseModel):
    interaction: Dict[str, Any]
    question: Dict[str, Any]


@app.get("/")
def root():
    return {"message": "Adaptive Q-learning backend running"}


@app.get("/test")
def test_database():
    response = {"backend": "ok", "db": False, "collections": []}
    try:
        response["db"] = db is not None
        if db is not None:
            response["collections"] = db.list_collection_names()
    except Exception as e:
        response["error"] = str(e)
    return response


@app.post("/api/generate-question", response_model=GenerateResponse)
def api_generate_question(req: GenerateRequest):
    profile = req.profile
    user_id = req.user_id

    # persist the latest profile snapshot (optional upsert)
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")

    state = profile_to_state(profile)
    decision = choose_action_db(user_id, state, profile)

    topic = (profile.get("topic") or decision["focusConcept"] or "arrays")

    # try Groq, fallback if unavailable
    used = "groq"
    try:
        payload = generate_with_groq(topic, decision["action"], decision["difficulty"], decision["focusConcept"])
    except Exception:
        used = "fallback"
        payload = fallback_question(topic, decision["action"], decision["difficulty"], decision["focusConcept"])

    # Save question document
    qdoc = Question(
        user_id=user_id,
        topic=str(topic),
        difficulty=decision["difficulty"],
        focus_concept=decision["focusConcept"],
        generator=used,
        title=payload.get("title"),
        description=payload.get("description"),
        starterCode=payload.get("starterCode"),
        solutionCode=payload.get("solutionCode"),
        expectedOutput=payload.get("expectedOutput"),
        testCases=payload.get("testCases") or [],
        tags=payload.get("tags") or [],
    )
    qid = create_document("question", qdoc)

    # also create a pending interaction row (without outcome yet)
    _ = create_document(
        "interaction",
        Interaction(
            user_id=user_id,
            question_id=qid,
            state=state,
            action=decision["action"],
            difficulty=decision["difficulty"],
            focus_concept=decision["focusConcept"],
        ),
    )

    return GenerateResponse(
        decision={"action": decision["action"], "difficulty": decision["difficulty"], "focusConcept": decision["focusConcept"]},
        state=state,
        question_id=qid,
        question=payload,
    )


@app.post("/api/submit-result")
def api_submit_result(req: SubmitRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")

    # reward shaping
    reward = 0.0
    if req.correct is True:
        reward += 1.0
    if req.correct is False:
        reward -= 0.5
    if req.rating is not None:
        reward += (float(req.rating) - 3.0) * 0.2  # [-0.4, +0.4]
    if req.time_ms is not None:
        # faster is better; cap at 5 minutes
        secs = min(float(req.time_ms) / 1000.0, 300.0)
        reward += (1.0 - secs / 300.0) * 0.3

    # Update interaction with outcome
    db["interaction"].update_one(
        {"user_id": req.user_id, "question_id": req.question_id},
        {"$set": {"correct": req.correct, "time_ms": req.time_ms, "rating": req.rating, "reward": reward}},
        upsert=True,
    )

    # Q-learning update
    alpha = float(os.getenv("Q_ALPHA", 0.3))
    gamma = float(os.getenv("Q_GAMMA", 0.9))

    # current Q(s,a)
    qdoc = db["qvalue"].find_one({"user_id": req.user_id, "state": req.state, "action": req.action})
    q_sa = float(qdoc.get("q", 0.0)) if qdoc else 0.0

    # next state and its max Q
    next_state = profile_to_state(req.next_profile) if req.next_profile else req.state
    next_qs = list(db["qvalue"].find({"user_id": req.user_id, "state": next_state}))
    max_next = max([float(x.get("q", 0.0)) for x in next_qs], default=0.0)

    target = reward + gamma * max_next
    new_q = (1 - alpha) * q_sa + alpha * target

    db["qvalue"].update_one(
        {"user_id": req.user_id, "state": req.state, "action": req.action},
        {"$set": {"q": new_q}},
        upsert=True,
    )

    # also store next_state on interaction
    db["interaction"].update_one(
        {"user_id": req.user_id, "question_id": req.question_id},
        {"$set": {"next_state": next_state}},
    )

    return {"ok": True, "reward": reward, "new_q": new_q, "next_state": next_state}


@app.get("/api/history", response_model=List[HistoryItem])
def api_history(user_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    inters = list(db["interaction"].find({"user_id": user_id}).sort("created_at", -1).limit(50))
    res: List[HistoryItem] = []
    for it in inters:
        qdoc = db["question"].find_one({"_id": it.get("question_id")})
        if qdoc is None:
            # lookup by stringified id stored earlier
            try:
                from bson import ObjectId
                qdoc = db["question"].find_one({"_id": ObjectId(str(it.get("question_id")))})
            except Exception:
                qdoc = None
        res.append(HistoryItem(interaction={k: v for k, v in it.items() if k != "_id"}, question={k: v for k, v in (qdoc or {}).items() if k != "_id"}))
    return res


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
