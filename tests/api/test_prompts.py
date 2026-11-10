"""Prompt sizing helpers."""

from __future__ import annotations

from corallab.api._prompts import TranscriptAssumption, count_tokens, synthetic_turns


def test_synthetic_turns_pair_up_and_can_leave_off_the_last_reply() -> None:
    a = TranscriptAssumption(user_tokens=5, assistant_tokens=10)
    full = synthetic_turns(3, a)
    assert [(t.turn_number, t.role) for t in full] == [
        (1, "user"),
        (1, "assistant"),
        (2, "user"),
        (2, "assistant"),
        (3, "user"),
        (3, "assistant"),
    ]
    assert synthetic_turns(3, a, final_reply=False)[-1].role == "user"
    assert len(full[0].content.split()) == 5


def test_token_count_scales_with_the_assumption() -> None:
    small = synthetic_turns(2, TranscriptAssumption(user_tokens=10, assistant_tokens=10))
    large = synthetic_turns(2, TranscriptAssumption(user_tokens=100, assistant_tokens=100))

    def count(turns: list) -> int:
        return count_tokens(
            "ollama_chat/llama3.2", [{"role": t.role, "content": t.content} for t in turns]
        )

    assert count(large) > count(small) * 5


def test_describe_names_both_sizes() -> None:
    assert TranscriptAssumption(1, 2).describe() == "~1 tokens per user message, ~2 per AI reply"
