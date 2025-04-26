# src/tools/write_tool.py
from crewai.tools import tool

@tool("write_field")
def write_field(field: str, value: str):
    """
    Write-back a single slot update.
    """
    # wb api call here
    return f"write-back queued: {field} = {value}"
