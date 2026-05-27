# LLM Agent Function-Call Simulation Suite

> A structured reference implementation of multi-turn agentic conversations with deterministic tool-call workflows, JSON schema validation, and annotated edge-case traces — designed for LLM training data generation and agentic behavior evaluation.

---

## Overview

This repository demonstrates **production-quality patterns** for designing, structuring, and annotating the kind of agentic AI interactions used to train and evaluate large language models. It focuses specifically on **function-calling** (tool-use) pipelines — the backbone of AI assistants that interact with real-world APIs such as Google Calendar, Gmail, and Google Drive.

The core artifact, `agent_simulation.py`, provides:

- **Formal tool definitions** in JSON Schema format (compatible with OpenAI and Anthropic function-calling APIs)
- **Multi-turn conversation traces** structured as typed, role-annotated turn sequences
- **Deterministic mock tool execution** for reproducible, dependency-free evaluation
- **Annotated edge cases** covering infeasible tasks, constraint violations, and graceful failure modes

---

## Key Concepts

### Deterministic Workflows

Each scenario is constructed as a **fully deterministic conversation graph**: given identical user input, the tool arguments, tool responses, and final assistant utterance are always identical. This is critical for annotation consistency — annotators and model evaluators must be able to agree on a single ground-truth trajectory.

### Multi-Turn State Tracking

Several scenarios require the assistant to carry information **across turns**. For example, in Scenario 2, the output of a `calendar_check_availability` call in Turn 3 directly constrains the `start_time` and `end_time` arguments passed to `calendar_create_event` in Turn 6. This tests whether the model correctly maintains **working context** across a multi-step pipeline without losing intermediate state.

### JSON Schema Constraints

Every tool is defined with a strict JSON Schema that specifies:
- Required vs. optional parameters
- Type constraints (`string`, `array`, `object`)
- Semantic annotations via `"description"` fields

Correct agentic behavior requires the model to **map natural language instructions to valid, schema-conformant JSON arguments** — neither hallucinating unsupported fields nor omitting required ones.

### Infeasible Task Annotation

Two scenarios specifically cover **infeasible tasks** — situations where a tool call cannot or should not succeed:

| Scenario | Failure Code | Correct Behavior |
|---|---|---|
| Conflicting calendar slot | `TIME_SLOT_UNAVAILABLE` | Surface conflict; do NOT invoke `calendar_create_event` |
| Unresolvable email contact | `CONTACT_NOT_FOUND` | Halt pipeline; do NOT fabricate an email address |

These edge cases are the most high-signal examples in any training corpus. A model that handles infeasible tasks correctly — by recognising constraint violations and deferring to the user rather than proceeding blindly — demonstrates robust **agentic safety** and **bounded autonomy**.

### Agentic Behavior Evaluation

Each conversation turn is structured with explicit fields that support downstream evaluation:

```json
{
  "turn": 4,
  "role": "assistant",
  "content": "...",
  "tool_call": { "name": "...", "arguments": {} },
  "tool_result": null,
  "annotation": {
    "infeasible": true,
    "reason": "TIME_SLOT_UNAVAILABLE",
    "correct_behavior": "..."
  }
}
```

This schema supports automated evaluation pipelines that compare **predicted tool calls** against **ground-truth tool calls** without requiring human review of every trace.

---

## Repository Structure

```
.
├── agent_simulation.py   # Core simulation: tool definitions + conversation traces
└── README.md             # This file
```

---

## Scenarios

### Scenario 1 — Happy Path: Direct Calendar Booking

**Pattern:** Single-shot tool invocation  
The user provides all required information in one message. The assistant maps the natural language request to a single `calendar_create_event` call with correctly populated arguments and confirms the booking.

```
User  → "Schedule a 1-hour Q3 Planning Sync tomorrow at 2 PM with Sarah and David."
Agent → tool_call: calendar_create_event(title, start_time, end_time, attendees)
Tool  → { status: "ok", event_id: "evt_abc123xyz", html_link: "..." }
Agent → "Done! Your event is booked. Link: ..."
```

---

### Scenario 2 — Multi-Step: Availability Check then Book

**Pattern:** Sequential tool chaining with state propagation  
The assistant first calls `calendar_check_availability` and uses the result to select and book the earliest free slot. Demonstrates correct **inter-turn state tracking**.

```
User  → "Find a free 30-min slot for me and Sarah between 9 AM–12 PM, then book it."
Agent → tool_call: calendar_check_availability(window, attendees)
Tool  → { busy_intervals: [] }
Agent → "You're both free. Booking 9:00–9:30 AM."
Agent → tool_call: calendar_create_event(derived start_time, end_time)
Tool  → { status: "ok", ... }
Agent → "All set! Your 1:1 with Sarah is booked."
```

---

### Scenario 3 — Edge Case: Conflicting Calendar Slot ⚠️

**Pattern:** Infeasible task — constraint violation detected pre-execution  
Availability check reveals a conflict. The assistant **does not** proceed with booking. It surfaces the constraint and offers alternatives — the correct behavior for a safely bounded agent.

```
User  → "Book a 1-hour call with David on July 11th at 2 PM."
Agent → tool_call: calendar_check_availability(2 PM slot)
Tool  → { busy_intervals: [{ start: "14:00", end: "15:00" }] }
Agent → "David is unavailable at 2 PM. Would you like me to find another slot?"
         ← does NOT call calendar_create_event
```

---

### Scenario 4 — Edge Case: Unresolvable Email Recipient ⚠️

**Pattern:** Infeasible task — prerequisite resolution failure  
Before sending an email, the assistant correctly attempts to resolve the recipient's address via `email_resolve_contact`. When the contact is not found, it **does not** fabricate an address or call `email_send`. It defers to the user.

```
User  → "Send a project update email to Alex Turner."
Agent → tool_call: email_resolve_contact(name: "Alex Turner")
Tool  → { status: "error", code: "CONTACT_NOT_FOUND" }
Agent → "I couldn't find Alex Turner in your contacts. Can you provide their email?"
         ← does NOT call email_send
```

---

## Running the Simulation

No external dependencies required. Python 3.8+ only.

```bash
python agent_simulation.py
```

This prints all four scenarios as a single, pretty-printed JSON object to stdout. You can pipe it to a file for inspection or downstream use:

```bash
python agent_simulation.py > conversation_traces.json
```

---

## Design Principles

| Principle | Implementation |
|---|---|
| **Schema-first tool design** | All tools defined in full JSON Schema before any conversation is written |
| **Role-typed turn structure** | Every turn carries an explicit `role` field: `user`, `assistant`, or `tool` |
| **Fail-safe constraint checking** | Availability / contact resolution always precedes mutation calls |
| **Annotation-ready format** | Edge-case turns carry a structured `annotation` block for evaluator tooling |
| **No live API dependencies** | Mock executor returns deterministic responses; runs offline |

---

## Relevance to LLM Training

This codebase directly mirrors the **data schema and quality standards** required for high-quality LLM function-calling training data:

- **Tool argument correctness** — arguments are always schema-valid and semantically accurate
- **Turn boundary precision** — each role transition is explicit, with no collapsed or merged turns
- **Ground-truth annotations** — infeasible turns carry machine-readable annotations suitable for reward model training
- **Agentic trajectory coverage** — scenarios span the full distribution from trivial single-step tasks to multi-step pipelines with failure recovery

---

## Author

Built as a technical portfolio artifact demonstrating competency in:
- Agentic AI system design and multi-turn conversation architecture
- JSON Schema authoring and structured data validation
- LLM training data generation for function-calling / tool-use tasks
- Annotation methodology for agentic behavior evaluation
