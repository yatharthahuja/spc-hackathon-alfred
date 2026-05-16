from models import SkillSpec

CATALOGUE: list[SkillSpec] = [
    SkillSpec(
        name="note",
        description="Save a text note to the notes file for the user.",
        parameters={"text": "The note content to persist."},
        examples=[
            "Take a note about the soup recipe",
            "Remember that dinner is at seven",
        ],
    ),
    SkillSpec(
        name="manipulate",
        description=(
            "Control the robot arm and kitchen layout. "
            "Use action reset to tidy the kitchen to normal."
        ),
        parameters={
            "action": "Either 'reset' (restore normalcy) or 'move'.",
            "target": "Location for move action, e.g. home, counter, drawer.",
        },
        examples=[
            "Tidy the kitchen",
            "Reset the counter",
            "Move arm to home",
        ],
    ),
]


def build_catalogue() -> list[SkillSpec]:
    return list(CATALOGUE)
