from __future__ import annotations


CHAT_ONLY_SKILLS = {
    "general_conversation",
    "answer_task_history",
    "describe_image_with_vlm",
    "speak",
}
ANSWER_SKILLS = {
    "answer_scene_question",
    "describe_image_with_vlm",
    "read_notes",
    "general_conversation",
    "answer_task_history",
}


def start_announcement(user_text: str, skill_name: str) -> str | None:
    normalized = _normalize(user_text)
    if skill_name in CHAT_ONLY_SKILLS:
        return None

    if skill_name == "pick_place_blue_marker":
        if _is_table_cleanup(normalized):
            return "Cleaning the table for you now, my friend."
        return "Moving the marker for you now, my friend."
    if skill_name == "pick_blue_marker":
        return "Picking up the marker for you now, my friend."
    if skill_name == "ordering":
        if "order" in normalized:
            return "Ordering that for you now, my friend."
        if "buy" in normalized or "purchase" in normalized:
            return "Getting that purchase started for you now, my friend."
        return "Getting that for you now, my friend."
    if skill_name == "read_notes":
        return "Reading your notes for you now."
    if skill_name == "save_notes":
        return "Saving your notes for you now."
    if skill_name in {"answer_scene_question", "capture_wrist_camera_image"}:
        return "Taking a look for you now."
    if skill_name == "go_home":
        return "Heading home now."
    if skill_name == "go_overlook":
        return "Moving to the overlook view now."
    if skill_name == "enzyme_experiments":
        return "Starting the experiment sequence for you now."

    return "I am starting that for you now."


def completion_announcement(
    user_text: str,
    skill_names: list[str],
    final_answer: str,
    has_error: bool,
) -> str | None:
    if has_error:
        return final_answer or "I ran into a problem while doing that."
    if not skill_names:
        return final_answer or None

    normalized = _normalize(user_text)
    primary_skill = skill_names[-1]
    if len(skill_names) == 1:
        primary_skill = skill_names[0]

    if primary_skill == "pick_place_blue_marker":
        if _is_table_cleanup(normalized):
            return "Table cleaned with pleasure."
        return "Marker moved with pleasure."
    if primary_skill == "pick_blue_marker":
        return "Marker picked up with pleasure."
    if primary_skill == "ordering":
        return "Done, I sent the order signal for you."
    if primary_skill == "save_notes":
        return "Done, I saved your notes."
    if primary_skill == "go_home":
        return "Home pose completed."
    if primary_skill == "go_overlook":
        return "Overlook view completed."
    if primary_skill == "enzyme_experiments":
        return "Experiment sequence completed with pleasure."
    if primary_skill in ANSWER_SKILLS:
        return final_answer or "Done, I completed that for you."
    if len(skill_names) > 1:
        if final_answer and final_answer != "Done.":
            return f"All set, I completed that for you. {final_answer}"
        return "All set, I completed that for you."

    if final_answer and final_answer != "Done.":
        return final_answer
    return "Done, I completed that for you."


def _is_table_cleanup(normalized: str) -> bool:
    cleanup_words = ("clean", "cleanup", "clear", "tidy", "remove")
    table_words = ("table", "desk", "workspace", "work surface")
    return any(word in normalized for word in cleanup_words) and any(
        word in normalized for word in table_words
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
