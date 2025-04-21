# src/tools/calendar_tool.py

from crewai.tools import BaseTool

class CalendarTool(BaseTool):
    # annotate these fields so Pydantic is happy
    name: str = "CalendarTool"
    description: str = "Books an interview given name, email, phone, datetime"

    def _run(self, full_name: str, email: str, phone: str, when: str) -> str:
        return f"✅ Meeting booked for {when}"
