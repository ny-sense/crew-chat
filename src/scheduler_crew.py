# src/scheduler_crew.py
import yaml
from crewai import Crew, Process, Agent
from crewai.project import CrewBase, agent, crew, task

from src.tools.calendar_tool import CalendarTool

# Load your YAML once at module load time
_AGENTS = yaml.safe_load(open("src/config/agents.yaml"))

@CrewBase
class SchedulerCrew:
    """A one‑agent, one‑task crew that collects any missing fields
       (full_name, email, phone, datetime) and then books the meeting."""
    agents_config = "config/agents.yaml"

    @agent
    def scheduler(self) -> Agent:
        cfg = _AGENTS["scheduler_agent"]
        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg.get(
                "backstory",
                "You are a friendly, efficient interview‐booking assistant."
            ),
            tools=[CalendarTool()],
            verbose=True,
        )

    @task
    def run_scheduler(self, inputs: dict = None) -> str:
        """
        Invoke the Scheduler agent to collect details and book the meeting.
        """
        # grab the last user message out of inputs

        inputs = inputs or {}
        msg = inputs.get("message", "")
        scheduler_agent = self.scheduler()
        return scheduler_agent.run(msg)

    @crew
    def crew(self) -> Crew:
        """
        Build and return the Crew instance with its agents and tasks.
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
