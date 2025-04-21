# src/chatbot_flow.py
from crewai.flow.flow import Flow, start, listen, router
from pydantic import BaseModel
from litellm import completion

from src.scheduler_crew import SchedulerCrew

SYSTEM_PROMPT = """
Execute these step by step.
Step 1. Prompt: `Hello! I'm a virtual recruiting assistant with Phillips Staffing. I'd love to help you out! Are you looking for job opportunities, or are you looking for talent`
Step 2. If the user is a Job Seeker : Jump to Step 4,
Step 3. If user is Business needing staffing support : Jump to Step 11

### For Job Seekers:
Step 4. Prompt: `Please provide the following details so we can assist you further:`
Step 5. Collect below details.
Step 6. Ask the user's Full Name.
Step 7. Ask the user's Phone number.
Step 8. Ask the user's Email address.
Step 9. Ask which branch location is closest to them. We have locations in (a) Spartanburg, (b) Easley, (c) Anderson, (d) Seneca, (e) Greenville, (f) Simpsonville. Show these locations in numbered format
Step 10. Book a meeting with recruiter.
Step 11. Prompt: `Awesome! We'll have a member of our team reach out, but in the meantime, you can take a look at our open positions here: https://jobs.phillipsstaffing.com/` - End.

### For Businesses:
Step 11. Prompt: `Glad to hear you're interested in partnering with Phillips Staffing! Can I get your name?`
Step 12. Ask the user's Company name.
Step 13. Ask the user's Email address.
Step 14. Ask the user's Phone number.
Step 15. Prompt: `That is all we needed! Thank you for your responses. A member from our sales team has received your information request and will be in contact with you shortly.` - End.

### Notifications
- **Job‑Seeker** → send email to `nitinyadav@sensehq.com` with subject `Website job seeker`, bulleted candidate info, then confirm jobs link.
- **Business** → send email with subject `Website Lead!`, bulleted lead info, then show “we only operate in Texas” if not TX.
- If email‑send fails, **do not** apologize or ask for confirmation—send it as soon as you have all answers.

**Tone & Validation**
- One question at a time, humble & respectful, 1–2 lines, emojis, no skips, basic validation on name/email/phone, only explain *why* if asked, call them by first name.
"""

class ChatState(BaseModel):
    history: list[dict] = []

class ChatbotFlow(Flow[ChatState]):

    #model = "gpt-4o-mini"

    model = "anthropic/claude-3-5-sonnet-20240620"

    def __init__(self):
        super().__init__()

    def kickoff(self, inputs: dict | None = None):
        # On first ever turn, inject your system prompt
        if not self.state.history:
            self.state.history.append({"role": "system", "content": SYSTEM_PROMPT})

        # Always record the user message
        if inputs and "message" in inputs:
            self.state.history.append({"role": "user", "content": inputs["message"]})

        # Delegate to @start()
        return super().kickoff()

    async def kickoff_async(self, inputs: dict | None = None):
        if not self.state.history:
            self.state.history.append({"role": "system", "content": SYSTEM_PROMPT})
        if inputs and "message" in inputs:
            self.state.history.append({"role": "user", "content": inputs["message"]})
        return await super().kickoff_async()

    @start()
    def receive(self) -> ChatState:
        # simply hand back the current (system + user) history
        return self.state

    @router(receive)
    def classify_intent(self, state: ChatState) -> str:
        last_msg = state.history[-1]["content"]
        # simple LLM‑based intent classifier (could be your own model)
        resp = completion(
            model=self.model,
            messages=[
                {"role": "system", "content":
                    "Classify the user’s message into exactly one of: "
                    "intent_schedule_interview, intent_get_job_match_info, or fallback."
                    "Respond with *only* the token (no extra text)."
                 },
                {"role": "user", "content": last_msg}
            ]
        )
        intent = resp["choices"][0]["message"]["content"].strip()
        print(f"detected intent- {intent}")
        if intent == "intent_schedule_interview":
            return "intent_schedule_interview"
        if intent == "intent_get_job_match_info":
            return "intent_get_job_match_info"
        return "fallback"  # anything else goes here

    @listen("intent_schedule_interview")
    def handoff_to_scheduler(self, state: ChatState) -> str:
        # instantiate & kickoff your SchedulerCrew
        scheduler = SchedulerCrew().crew()
        # pass in the last user message as the “prompt” for that crew
        crew_output = scheduler.kickoff(inputs={"message": state.history[-1]["content"]})

        reply_str = crew_output.raw

        # record and return
        state.history.append({"role": "assistant", "content": reply_str})
        return reply_str

    @listen("intent_get_job_match_info")
    def handle_job_info(self, state: ChatState) -> str:
        # TODO: invoke job‐matching tool/API here
        return (
            "Here are a few roles that match your profile:\n"
            "• Position A\n• Position B\n• Position C\n"
            "Feel free to click through for more details!"
        )

    @listen("fallback")
    def reply(self, state: ChatState) -> str:
        # if no special intent, fall back to your original LLM reply
        resp = completion(model=self.model, messages=state.history)
        text = resp["choices"][0]["message"]["content"]
        self.state.history.append({"role": "assistant", "content": text})
        return text

    @listen(reply)
    def detect_intent(self, last_reply: str) -> str:
        # placeholder for your future intent logic:
        #    intent = classify(last_user_message, self.state.history)
        #    if intent == "some_action": ...
        #
        # For now, just return the assistant’s reply so kickoff() stays a string:
        return last_reply
