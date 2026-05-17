from __future__ import annotations

from app.orchestrator.announcements import completion_announcement, start_announcement


def test_clean_table_announcements_are_natural():
    user_text = "clean the table"

    assert start_announcement(user_text, "pick_place_blue_marker") == (
        "Cleaning the table for you now, my friend."
    )
    assert completion_announcement(
        user_text,
        ["pick_place_blue_marker"],
        "Picked up and placed the blue marker.",
        has_error=False,
    ) == "Table cleaned with pleasure."


def test_ordering_announcements_are_natural():
    user_text = "can you please get apple and oranges for me"

    assert start_announcement(user_text, "ordering") == "Getting that for you now, my friend."
    assert completion_announcement(
        user_text,
        ["ordering"],
        "I sent the order signal.",
        has_error=False,
    ) == "Done, I sent the order signal for you."


def test_chat_skills_do_not_need_pre_announcement():
    assert start_announcement("hi Alfred", "general_conversation") is None
