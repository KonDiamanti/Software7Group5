from app.db import Database
from app.guard import Guard, GuardResult
from app.models import EchoModel
from app.pipeline import BLOCKED_MESSAGE, Pipeline


class BlockWord(Guard):
    """Test guard that blocks any text containing 'forbidden'."""

    def check(self, text, direction):
        if "forbidden" in text:
            return GuardResult(False, "contains forbidden word", 1.0)
        return GuardResult(True, "clean", 0.0)


def make_pipeline(guard):
    db = Database(":memory:")
    db.init_schema()
    return Pipeline(EchoModel("test"), guard, db), db


def test_allowed_prompt_gets_reply():
    pipeline, db = make_pipeline(BlockWord("test"))
    result = pipeline.run("hello")
    assert not result.blocked
    assert result.reply == "[echo] hello"
    assert len(db.get_messages(result.conversation_id)) == 2
    assert len(db.get_guard_logs(result.conversation_id)) == 2  # input + output


def test_blocked_input_never_reaches_model():
    pipeline, db = make_pipeline(BlockWord("test"))
    result = pipeline.run("this is forbidden")
    assert result.blocked and result.blocked_at == "input"
    assert result.reply == BLOCKED_MESSAGE
    messages = db.get_messages(result.conversation_id)
    assert messages[-1]["content"] == BLOCKED_MESSAGE
