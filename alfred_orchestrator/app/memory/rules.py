SAFETY_RULES = [
    "The LLM may only call registered skills.",
    "The LLM may not generate raw motor commands.",
    "Physical skills must be scripted and bounded.",
    "Any arm motion must use predefined safe poses.",
    "If confidence is low, ask the user or retry vision.",
    "If emergency stop is active, reject all physical skills.",
    "Do not order, purchase, or send external messages without explicit confirmation.",
]
