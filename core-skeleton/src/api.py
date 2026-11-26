# api.py
# the web front door of the service. Apps send prompts to /check, report missed attacks to /feedback.
#
#   POST /check     {"app": "chatbot", "text": "..."}  ->  block or pass, and what decided it
#   POST /feedback  {"app": "chatbot", "text": "..."}  ->  stores a missed attack in the shared database
#   GET  /stats     how each app's prompts were decided (memory, tier 1 or guard)
#   GET  /health    is the service running, how many attacks are stored
#
# With share=True all apps use one attack database and one tier 1 (the network effect).
# With share=False every app gets its own, as if each app ran the check alone.
#
# The endpoints are async on purpose: sqlite connections may only be used in the thread
# that made them, and async endpoints all run in the one thread of the event loop.
# The services are made on first use, inside that thread.

import logging
from collections import Counter
from dataclasses import dataclass, field

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.result import BLOCK
from src.service import PromptService

log = logging.getLogger("sw7.api")


@dataclass
class AppState:
    service: PromptService
    db: object
    tier1: object
    guard: object
    stats: Counter = field(default_factory=Counter)


class Services:
    """One PromptService per app. make_db(), make_tier1() and make_guard(app_name) build the parts."""

    def __init__(self, make_db, make_tier1, make_guard, share=True):
        self.make_db = make_db
        self.make_tier1 = make_tier1
        self.make_guard = make_guard
        self.share = share
        self.apps = {}
        self._shared = None

    def for_app(self, name):
        if name not in self.apps:
            if self.share:
                if self._shared is None:
                    self._shared = (self.make_db(), self.make_tier1())
                db, tier1 = self._shared
            else:
                db, tier1 = self.make_db(), self.make_tier1()
            guard = self.make_guard(name)
            # a tier 1 that can learn (LearningTier1) gets every guard answer
            callback = getattr(tier1, "on_guard_answer", None)
            service = PromptService(db, tier1, guard, on_guard_answer=callback, app_name=name)
            self.apps[name] = AppState(service, db, tier1, guard)
        return self.apps[name]

    def attacks_stored(self):
        databases = {id(state.db): state.db for state in self.apps.values()}
        return sum(db.count() for db in databases.values())


def decided_by(reason):
    if reason == "known_attack":
        return "memory"
    if reason in ("guard_block", "guard_pass"):
        return "guard"
    return "tier1"


class CheckRequest(BaseModel):
    app: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=20000)


class CheckResponse(BaseModel):
    verdict: str
    reason: str
    score: float | None = None
    decided_by: str


def create_app(services: Services) -> FastAPI:
    api = FastAPI(title="Shared prompt check")

    @api.post("/check", response_model=CheckResponse)
    async def check(request: CheckRequest):
        state = services.for_app(request.app)
        try:
            result = state.service.check(request.text)
            verdict, reason, score = result.verdict, result.reason, result.score
        except Exception:
            # if something breaks (for example the guard is down) we fail closed: block
            log.exception("check failed")
            verdict, reason, score = BLOCK, "error", None
        by = "error" if reason == "error" else decided_by(reason)
        state.stats[by] += 1
        state.stats[verdict] += 1
        return CheckResponse(verdict=verdict, reason=reason, score=score, decided_by=by)

    @api.post("/feedback")
    async def feedback(request: CheckRequest):
        # an app reports an attack that got through
        state = services.for_app(request.app)
        state.db.add(request.text, source=f"feedback:{request.app}")
        learn = getattr(state.tier1, "learn", None)
        if learn is not None:
            learn(request.text, True)
        return {"status": "stored", "attacks_stored": services.attacks_stored()}

    @api.get("/stats")
    async def stats():
        return {name: dict(state.stats) for name, state in services.apps.items()}

    @api.get("/health")
    async def health():
        return {"status": "ok", "share": services.share, "apps": len(services.apps),
                "attacks_stored": services.attacks_stored()}

    return api
