from app.db import Database


def test_schema_and_message_roundtrip():
    db = Database(":memory:")
    db.init_schema()
    conv = db.create_conversation()
    db.add_message(conv, "user", "hi")
    db.add_message(conv, "assistant", "hello", "placeholder")
    rows = db.get_messages(conv)
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[1]["model_name"] == "placeholder"


def test_guard_log_roundtrip():
    db = Database(":memory:")
    db.init_schema()
    conv = db.create_conversation()
    db.add_guard_log(conv, "input", "g", True, "ok", 0.1)
    logs = db.get_guard_logs(conv)
    assert len(logs) == 1
    assert logs[0]["allowed"] == 1
