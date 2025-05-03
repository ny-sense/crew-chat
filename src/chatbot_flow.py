# src/chatbot_flow.py
import json
import re
import asyncio

from crewai.flow.flow import Flow, start, listen, router
from pydantic import BaseModel
from litellm import completion
from src.tools.scheduler_tool import schedule_interview

from src.scheduler_crew import SchedulerCrew
from src.tools.write_tool import write_field
from src.decorator import timeit_listener

TOOLS = {
    "schedule_interview": schedule_interview,
    # "send_email":         send_email,
    "write_field": write_field,
}

ALLOWED_KEYS = [
    "first_name",
    "last_name",
    "full_name",
    "email",
    "phone",
    "branch_location",
    "looking_for",
]

SYSTEM_PROMPT_USER_INTENT = """Detect if user's utterance is a frequently asked question. At each user turn,
classify the user’s message into exactly one of:
 
 • intent_faq                  (they’re asking a frequently‑asked question)
 • intent_file_upload          (they want to upload a file)  
 • fallback                    (anything else) 

If you choose intent_faq, do _not_ ask for more data—you’ll handle it separately."""

SYSTEM_PROMPT = """
Execute these step by step.
Step 1. Prompt: `Hello! I'm a virtual recruiting assistant with Sense.`
Step 2. If job_role == "nurse" : Jump to Step 3, else jump to Step 10

Step 4. Prompt: `It seems you are looking for {{job_role}} job.`
Step 5. Collect below details.
Step 6. Ask the user's Full Name.
Step 7. Ask the user's Phone number.
Step 8. Ask the user's Email address.
Step 9. Ask which branch location is closest to them. We have locations in (a) Spartanburg, (b) Easley, (c) Anderson, (d) Seneca, (e) Greenville, (f) Simpsonville
Step 10. Emit exactly `TOOL_CALL schedule_interview`.
Step 11. Prompt: `Awesome! We'll have a member of our team reach out, thank you!` - End.

Step 11. Prompt: `Glad to hear you're working as {{job_role}}! Can I get your name?`
Step 12. Ask where user is working right now.
Step 13. Ask his shift details.
Step 14. Ask how much pay he is looking for per hour.
Step 15. Prompt: `That is all we needed! Thank you for your responses.` - End.

**Tone & Validation**
- One question at a time, 
- Be humble & respectful, 1–2 lines, emojis, no skips, basic validation on name/email/phone, only explain *why* if asked, call them by first name.
"""

class ChatState(BaseModel):
    history: list[dict] = []
    context: dict = {}

class ChatbotFlow(Flow[ChatState]):

    #model = "gpt-4o-mini"

    model = "anthropic/claude-3-5-sonnet-20240620"

    def __init__(self):
        super().__init__()

    def kickoff(self, inputs: dict | None = None):
        # On first ever turn, inject your system prompt
        if not self.state.history:
            ctx = json.dumps(self.state.context, indent=2)
            print(f"ctx: {ctx}")
            self.state.history.append({"role": "system", "content": SYSTEM_PROMPT + "\n\n# CONTEXT:\n" + ctx})

        # Always record the user message
        if inputs and "message" in inputs:
            self.state.history.append({"role": "user", "content": inputs["message"]})
            # accumulate context however you like:
            self.state.context.update(inputs.get("context", {}))

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

    @listen(receive)
    @timeit_listener
    async def extract_info(self, state: ChatState) -> None:
        # only extract when the turn before last was the bot asking something
        if len(state.history) < 2 or state.history[-2]["role"] != "assistant":
            return None

        question = state.history[-2]["content"]
        answer = state.history[-1]["content"]

        print(f"question: {question}")
        print(f"answer: {answer}")

        # build the bullet list from ALLOWED_KEYS
        bullets = "\n".join(f"• {k}" for k in ALLOWED_KEYS)

        system_prompt = f"""
        You’re a JSON extractor.  You will be given:

          1) A **Question** (so you know what data to look for)  
          2) A **User’s Answer** (the only place you should pull all values from)  

        **INSTRUCTIONS**  
        - **Only** pull values that actually appear in the Answer text.
        - Extract **all** fields related to the Answer text.
        - Do *not* include any explanatory text.  
        - Do *not* wrap your JSON in markdown fences (```), code blocks, or quotes.  
        - Do *not* add any keys other than the allowed ones.  
        - Do *not* add any whitespace or punctuation before or after the JSON.  
        - Do *not* extract any values from the Question itself—use it only to understand which fields to look for.  

        Allowed keys (exact spelling):
        {bullets}

        Return ** a JSON object ** mapping each found key to its value.  
        If you find none, return {{}}.
        """

        #print(system_prompt)

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":
                f"Question:\n{question}\n\n"
                f"Answer:\n{answer}"
             }
        ]

        # a single blocking call here; we can offload to a thread
        resp = completion(model=self.model, messages=prompt_messages)

        print(f"resp: {resp}")
        # resp = completion(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        # )

        raw = resp["choices"][0]["message"]["content"].strip()
        print(f"raw: {raw}")

        # be permissive about backticks or code fences
        raw = re.sub(r"^```json|```$", "", raw, flags=re.I).strip()
        print(f"raw: {raw}")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # if parsing fails, just skip
            return None

        print(f"extracted json: {data}")

        # Keep only allowed keys
        filtered = {k: v for k, v in data.items() if k in ALLOWED_KEYS}

        # If none of the keys were allowed, do nothing
        if not filtered:
            return None
        print(f"filtered items: {filtered}")
        for field, val in filtered.items():
            # merge whatever got extracted
            state.context[field] = val

            # Call the write_field tool
            # out = TOOLS["write_field"].run( field, val)
            out = asyncio.create_task(
                self._run_write_tool(field, val, state)
            )
            print(f"{out}")

        print(f"state.context = {state.context}")
        return None

    async def _run_write_tool(self, field: str, value: str, state: ChatState):
        """
        Call write_field tool asynchronously.
        """
        result = write_field.run(field,value)
        # if it returns a coroutine, await it
        if asyncio.iscoroutine(result):
            result = await result
        print(result)
        return result

    @router(receive)
    @timeit_listener
    def classify_intent(self, state: ChatState) -> str:

        # grab the last assistant turn (or empty if there isn’t one)
        last_assistant_msg = next(
            (m["content"] for m in reversed(state.history) if m["role"] == "assistant"),
            ""
        )

        last_user_msg = state.history[-1]["content"]

        # few‑shot examples to help the LLM
        examples = [
            ("How do I know I got the job?", "intent_faq"),
            ("What positions do you have?", "intent_faq"),
            ("Any work from home positions?", "intent_faq"),
            ("I want to see my paycheck stub", "intent_faq"),
            ("Do you have a Sales position open for 23227 Richmond va", "intent_faq"),
            ("Is this a remote , hybrid or onsite role ", "intent_faq"),
            ("I want to upload my resume", "intent_file_upload")
        ]
        # build our little classification prompt
        prompt = f"""
            You are a router that decides whether the *user’s latest message* is:
              • A reply to the *previous assistant question*, or
              • A brand-new FAQ request.
            
            Here’s the last assistant question:
            \"\"\"{last_assistant_msg}\"\"\"
            
            And here’s the user’s reply:
            \"\"\"{last_user_msg}\"\"\"
            
            Classify this into exactly one of:
              • intent_answer — meaning “this answers my previous question”
              • intent_faq — meaning “this is a brand-new FAQ”
              • intent_schedule_interview
              • intent_get_job_match_info
            
            Respond with exactly the label.
            **Greeting messages e.g. Hello, Hola are not FAQS**
            """
        for msg, label in examples:
            prompt += f"User: \"{msg}\"\nLabel: {label}\n\n"
        prompt += f"User: \"{last_user_msg}\"\nLabel:"

        resp = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        intent = resp["choices"][0]["message"]["content"].strip()
        print(f"detected intent- {intent}")
        if intent == "intent_faq":
            return intent
        else:
            return "fallback"  # anything else goes here

    @listen("intent_faq")
    @timeit_listener
    def handle_faq(self, state: ChatState) -> str:
        # Here you can either embed a small FAQ lookup,
        # hit an external FAQ‐tool, or just ask the LLM to answer:

        last_user_msg = state.history[-1]["content"]

        resp = completion(
            model=self.model,
            messages=state.history + [
                {"role": "system", "content":
                    f"Answer the following question as a Phillips Staffing FAQ.  "
                    f""
                    f"{last_user_msg}"
                    f"Be concise, 1–2 sentences."}
            ]
        )
        answer = resp["choices"][0]["message"]["content"]
        state.history.append({"role": "assistant", "content": answer})
        print(f"faq answer: {answer}")
        return answer

    @listen("intent_schedule_interview")
    @timeit_listener
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
    @timeit_listener
    def reply(self, state: ChatState) -> str:
        # if no special intent, fall back to your original LLM reply
        resp = completion(model=self.model, messages=state.history)
        text = resp["choices"][0]["message"]["content"]
        print(f"response text: {text}")
        self.state.history.append({"role": "assistant", "content": text})

        # Look for *any* TOOL_CALL
        m = re.search(
            r"(.*?)[\r\n]*TOOL_CALL\s+(\w+)[\r\n]*(.*)",
            text,
            re.S
        )
        if m:
            prefix, tool_name, suffix = m.group(1).strip(), m.group(2), m.group(3).strip()

            # Find & invoke the tool
            fn = TOOLS.get(tool_name)
            if not fn:
                err = f"Unknown tool `{tool_name}`."
                state.history.append({"role": "tool", "content": err})
                return err

            result = fn.run(state.context)
            state.history.append({"role": "tool", "content": result})

            # Stitch prefix, tool result, and suffix back together
            parts = [p for p in (prefix, result, suffix) if p]
            return "\n\n".join(parts)

        return text

    @listen(reply)
    @timeit_listener
    def detect_intent(self, last_reply: str) -> str:
        # placeholder for future intent logic:
        #    intent = classify(last_user_message, self.state.history)
        #    if intent == "some_action": ...
        #
        # For now, just return the assistant’s reply so kickoff() stays a string:
        return last_reply
