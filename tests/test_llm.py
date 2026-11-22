from app.llm import to_bedrock


def test_system_messages_are_split_out():
    system, convo = to_bedrock(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ]
    )
    assert system == [{"text": "Be brief."}]
    assert convo == [{"role": "user", "content": [{"text": "Hi"}]}]
