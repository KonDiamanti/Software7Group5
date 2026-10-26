"""Guard -> model -> guard flow, with everything logged to the database."""

from __future__ import annotations

from dataclasses import dataclass

from .db import Database
from .guard import Guard
from .models import LanguageModel

BLOCKED_MESSAGE = "Sorry, I can't help with that request."


@dataclass
class PipelineResult:
    conversation_id: int
    reply: str
    blocked: bool
    blocked_at: str | None = None  # "input" or "output"


class Pipeline:
    def __init__(
        self,
        model: LanguageModel,
        guard: Guard,
        db: Database,
        check_input: bool = True,
        check_output: bool = True,
    ):
        self.model = model
        self.guard = guard
        self.db = db
        self.check_input = check_input
        self.check_output = check_output

    def run(self, prompt: str, conversation_id: int | None = None) -> PipelineResult:
        conv_id = conversation_id or self.db.create_conversation()
        user_msg_id = self.db.add_message(conv_id, "user", prompt)

        if self.check_input:
            verdict = self.guard.check(prompt, "input")
            self.db.add_guard_log(
                conv_id, "input", self.guard.name, verdict.allowed,
                verdict.reason, verdict.score, user_msg_id,
            )
            if not verdict.allowed:
                self.db.add_message(conv_id, "assistant", BLOCKED_MESSAGE)
                return PipelineResult(conv_id, BLOCKED_MESSAGE, True, "input")

        reply = self.model.generate(prompt)

        if self.check_output:
            verdict = self.guard.check(reply, "output")
            if not verdict.allowed:
                self.db.add_guard_log(
                    conv_id, "output", self.guard.name, False,
                    verdict.reason, verdict.score, user_msg_id,
                )
                self.db.add_message(conv_id, "assistant", BLOCKED_MESSAGE)
                return PipelineResult(conv_id, BLOCKED_MESSAGE, True, "output")

        reply_msg_id = self.db.add_message(conv_id, "assistant", reply, self.model.name)
        if self.check_output:
            self.db.add_guard_log(
                conv_id, "output", self.guard.name, True,
                verdict.reason, verdict.score, reply_msg_id,
            )
        return PipelineResult(conv_id, reply, False)
