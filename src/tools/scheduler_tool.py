# src/tools/scheduler_tool.py
from crewai.tools import tool
from typing import Any, Dict

@tool("schedule_interview")
def schedule_interview(context: Dict[str, Any]) -> str:
    """
    context contains everything the user has told us so far.
    Here you can:
      1. Inspect context for full_name, email, phone.
      2. Ask follow‑ups if something’s missing (by calling LLM again).
      3. Call your real calendar API when ready.
    Return the final confirmation string.
    """
    name = context.get("full_name", "<no-name>")
    email = context.get("email", "<no-email>")
    phone = context.get("phone", "<no-phone>")
    # TODO: fill in missing‑info loop if you like
    return f"Interview booked for {name} ({email}, {phone})!"
