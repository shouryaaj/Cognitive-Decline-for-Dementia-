from longitudinal.session_store import save_session, load_user_history

def test_store_and_load():
    save_session({
        "user_id": "TEST_USER",
        "timestamp": "2025-01-01",
        "chi": 65,
        "memory": 0.2,
        "attention": -0.1,
        "speed": 0.3,
        "errors": 1
    })

    history = load_user_history("TEST_USER")
    assert len(history) >= 1
