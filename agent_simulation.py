"""
agent_simulation.py
====================
Demonstrates structured multi-turn agentic conversations with mock tool/function
call definitions for Google Calendar and Email APIs.

Each scenario is represented as an ordered list of conversation turns, where each
turn is a typed dict carrying: role, content, tool_call (optional), and tool_result
(optional). This mirrors the schema used by OpenAI / Anthropic function-calling APIs
and is suitable for LLM trainer annotation and agentic behavior evaluation.

Scenarios covered
-----------------
1. Happy path  – Schedule a meeting via Calendar API
2. Multi-step  – Find a free slot, then create an event (state carries across turns)
3. Edge case   – Infeasible task: attempt to book a slot that is already occupied
4. Edge case   – Draft an email to an unknown / unresolvable recipient

Author : <Faruk Alimi>
Date   : 2025
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# 1. TOOL DEFINITIONS  (JSON-Schema style, mirrors OpenAI / Anthropic format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "calendar_create_event",
        "description": (
            "Creates a new event on the user's Google Calendar. "
            "Returns the created event object or an error if the slot is unavailable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title":      {"type": "string",  "description": "Event title / summary."},
                "start_time": {"type": "string",  "description": "ISO-8601 datetime string (e.g. 2025-07-10T14:00:00)."},
                "end_time":   {"type": "string",  "description": "ISO-8601 datetime string."},
                "attendees":  {"type": "array",   "items": {"type": "string"}, "description": "List of attendee email addresses."},
                "description":{"type": "string",  "description": "Optional event description."},
            },
            "required": ["title", "start_time", "end_time"],
        },
    },
    {
        "name": "calendar_check_availability",
        "description": (
            "Checks whether the user (and optionally a set of attendees) are free "
            "within a given time window. Returns a list of busy intervals."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {"type": "string",  "description": "Start of the window (ISO-8601)."},
                "end_time":   {"type": "string",  "description": "End of the window (ISO-8601)."},
                "attendees":  {"type": "array",   "items": {"type": "string"}, "description": "Emails to check."},
            },
            "required": ["start_time", "end_time"],
        },
    },
    {
        "name": "email_send",
        "description": "Sends an email on behalf of the user via the Gmail API.",
        "parameters": {
            "type": "object",
            "properties": {
                "to":      {"type": "array",  "items": {"type": "string"}, "description": "Recipient email addresses."},
                "subject": {"type": "string", "description": "Email subject line."},
                "body":    {"type": "string", "description": "Plain-text email body."},
                "cc":      {"type": "array",  "items": {"type": "string"}, "description": "Optional CC list."},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "email_resolve_contact",
        "description": (
            "Looks up a contact by name and returns their email address. "
            "Returns null if the contact cannot be resolved."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full or partial display name of the contact."},
            },
            "required": ["name"],
        },
    },
]


# ---------------------------------------------------------------------------
# 2. MOCK TOOL EXECUTOR  (simulates API responses without a live network call)
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Simulates the execution of a tool and returns a structured mock response.

    In a production system this function would dispatch to real API clients.
    Here we return deterministic mock data so the conversation flow can be
    evaluated without external dependencies.

    Parameters
    ----------
    tool_name : str
        Name of the tool to execute (must match a key in TOOL_DEFINITIONS).
    arguments : dict
        Validated arguments for the tool call.

    Returns
    -------
    dict
        A structured tool result with either a ``data`` payload or an
        ``error`` object containing a machine-readable ``code`` and a
        human-readable ``message``.
    """

    # ---- calendar_check_availability ----------------------------------------
    if tool_name == "calendar_check_availability":
        # Scenario 1 & 2: return an empty busy list (everyone is free)
        if arguments.get("start_time", "").startswith("2025-07-10T09"):
            return {
                "status": "ok",
                "busy_intervals": [],
                "message": "No conflicts detected in the requested window.",
            }
        # Scenario 3: the slot is already taken – simulate a conflict
        if arguments.get("start_time", "").startswith("2025-07-11T14"):
            return {
                "status": "ok",
                "busy_intervals": [
                    {"start": "2025-07-11T14:00:00", "end": "2025-07-11T15:00:00"},
                ],
                "message": "One or more attendees have a conflict in this window.",
            }

    # ---- calendar_create_event ----------------------------------------------
    elif tool_name == "calendar_create_event":
        # Reject if the slot we know is blocked is requested (defensive check)
        if arguments.get("start_time", "").startswith("2025-07-11T14"):
            return {
                "status": "error",
                "error": {
                    "code": "TIME_SLOT_UNAVAILABLE",
                    "message": (
                        "The requested time slot (2025-07-11 14:00–15:00) overlaps "
                        "with an existing event. Please choose a different time."
                    ),
                },
            }
        return {
            "status": "ok",
            "data": {
                "event_id":   "evt_abc123xyz",
                "title":      arguments.get("title"),
                "start_time": arguments.get("start_time"),
                "end_time":   arguments.get("end_time"),
                "attendees":  arguments.get("attendees", []),
                "html_link":  "https://calendar.google.com/event?eid=evt_abc123xyz",
            },
        }

    # ---- email_resolve_contact -----------------------------------------------
    elif tool_name == "email_resolve_contact":
        known_contacts = {
            "sarah": "sarah.jones@example.com",
            "sarah jones": "sarah.jones@example.com",
            "david": "david.kim@example.com",
        }
        name_key = arguments.get("name", "").lower()
        email = known_contacts.get(name_key)
        if email:
            return {"status": "ok", "data": {"name": arguments["name"], "email": email}}
        # Edge case: contact not found
        return {
            "status": "error",
            "error": {
                "code": "CONTACT_NOT_FOUND",
                "message": (
                    f"No contact matching '{arguments.get('name')}' was found in "
                    "the user's directory. Cannot resolve email address."
                ),
            },
        }

    # ---- email_send ---------------------------------------------------------
    elif tool_name == "email_send":
        return {
            "status": "ok",
            "data": {
                "message_id": "msg_789qrs",
                "to":         arguments.get("to"),
                "subject":    arguments.get("subject"),
                "sent_at":    "2025-07-10T08:35:00Z",
            },
        }

    # Unknown tool — should never happen in a well-constrained system
    return {
        "status": "error",
        "error": {
            "code": "UNKNOWN_TOOL",
            "message": f"Tool '{tool_name}' is not registered.",
        },
    }


# ---------------------------------------------------------------------------
# 3. CONVERSATION SCHEMAS  (multi-turn structured traces)
# ---------------------------------------------------------------------------
# Each turn is a dict with:
#   role        : "user" | "assistant" | "tool"
#   content     : natural-language text (None for pure tool-call turns)
#   tool_call   : {name, arguments} – present when the assistant invokes a tool
#   tool_result : the structured response returned by execute_tool()
# ---------------------------------------------------------------------------

def build_scenario_1() -> list[dict[str, Any]]:
    """
    Scenario 1 – Happy Path: Schedule a team meeting directly.

    The user provides all necessary information in a single prompt, so the
    assistant can invoke `calendar_create_event` without any clarification turns.
    This tests single-shot tool invocation and correct argument mapping.
    """
    tool_args = {
        "title":       "Q3 Planning Sync",
        "start_time":  "2025-07-10T14:00:00",
        "end_time":    "2025-07-10T15:00:00",
        "attendees":   ["sarah.jones@example.com", "david.kim@example.com"],
        "description": "Quarterly planning session to align on OKRs.",
    }
    tool_result = execute_tool("calendar_create_event", tool_args)

    return [
        {
            "turn": 1,
            "role": "user",
            "content": (
                "Schedule a 1-hour team meeting called 'Q3 Planning Sync' for "
                "tomorrow at 2 PM with sarah.jones@example.com and "
                "david.kim@example.com. Add a note: 'Quarterly planning session "
                "to align on OKRs.'"
            ),
            "tool_call":   None,
            "tool_result": None,
        },
        {
            "turn": 2,
            "role": "assistant",
            "content": None,  # pure tool-dispatch turn; no user-visible text yet
            "tool_call": {
                "name":      "calendar_create_event",
                "arguments": tool_args,
            },
            "tool_result": None,
        },
        {
            "turn": 3,
            "role": "tool",
            "content": None,
            "tool_call":   None,
            "tool_result": tool_result,
        },
        {
            "turn": 4,
            "role": "assistant",
            "content": (
                "Done! I've created the 'Q3 Planning Sync' event for July 10, 2025 "
                "from 2:00 PM to 3:00 PM. Sarah Jones and David Kim have been added "
                "as attendees. You can view or edit the event here: "
                "https://calendar.google.com/event?eid=evt_abc123xyz"
            ),
            "tool_call":   None,
            "tool_result": None,
        },
    ]


def build_scenario_2() -> list[dict[str, Any]]:
    """
    Scenario 2 – Multi-Step / State Tracking: Check availability, then book.

    The user asks the assistant to find a free slot *before* creating the event.
    This tests multi-turn state tracking: the result of `calendar_check_availability`
    (turn 3) must inform the arguments of `calendar_create_event` (turn 6).
    """
    avail_args   = {"start_time": "2025-07-10T09:00:00", "end_time": "2025-07-10T12:00:00",
                    "attendees": ["sarah.jones@example.com"]}
    avail_result = execute_tool("calendar_check_availability", avail_args)

    create_args   = {"title": "1:1 with Sarah", "start_time": "2025-07-10T09:00:00",
                     "end_time": "2025-07-10T09:30:00", "attendees": ["sarah.jones@example.com"]}
    create_result = execute_tool("calendar_create_event", create_args)

    return [
        {
            "turn": 1,
            "role": "user",
            "content": (
                "I want to set up a 30-minute 1:1 with Sarah Jones sometime between "
                "9 AM and 12 PM tomorrow. First check if we're both free, then book "
                "the earliest available slot."
            ),
            "tool_call":   None,
            "tool_result": None,
        },
        {
            "turn": 2,
            "role": "assistant",
            "content": "I'll check availability for you and Sarah between 9 AM and 12 PM first.",
            "tool_call": {
                "name":      "calendar_check_availability",
                "arguments": avail_args,
            },
            "tool_result": None,
        },
        {
            "turn": 3,
            "role": "tool",
            "content":     None,
            "tool_call":   None,
            "tool_result": avail_result,
        },
        {
            "turn": 4,
            "role": "assistant",
            "content": (
                "Great news — you and Sarah are both free from 9:00 AM. "
                "I'll book 9:00–9:30 AM for your 1:1 now."
            ),
            "tool_call": {
                "name":      "calendar_create_event",
                "arguments": create_args,
            },
            "tool_result": None,
        },
        {
            "turn": 5,
            "role": "tool",
            "content":     None,
            "tool_call":   None,
            "tool_result": create_result,
        },
        {
            "turn": 6,
            "role": "assistant",
            "content": (
                "All set! Your 1:1 with Sarah Jones is booked for July 10, 2025 "
                "from 9:00 AM to 9:30 AM. "
                "Link: https://calendar.google.com/event?eid=evt_abc123xyz"
            ),
            "tool_call":   None,
            "tool_result": None,
        },
    ]


def build_scenario_3() -> list[dict[str, Any]]:
    """
    Scenario 3 – Infeasible Task (Edge Case): Booking a blocked time slot.

    The assistant first checks availability and discovers a conflict. Rather than
    proceeding blindly, it surfaces the constraint to the user and requests
    clarification — demonstrating safe, constraint-aware agentic behavior.

    This is the canonical 'infeasible' pattern expected in LLM trainer annotations:
    the model must recognise an insurmountable constraint and fail gracefully.
    """
    avail_args   = {"start_time": "2025-07-11T14:00:00", "end_time": "2025-07-11T15:00:00",
                    "attendees": ["david.kim@example.com"]}
    avail_result = execute_tool("calendar_check_availability", avail_args)

    return [
        {
            "turn": 1,
            "role": "user",
            "content": (
                "Book a 1-hour call with David Kim on July 11th at 2 PM."
            ),
            "tool_call":   None,
            "tool_result": None,
        },
        {
            "turn": 2,
            "role": "assistant",
            "content": "Let me check David's availability at that time before booking.",
            "tool_call": {
                "name":      "calendar_check_availability",
                "arguments": avail_args,
            },
            "tool_result": None,
        },
        {
            "turn": 3,
            "role": "tool",
            "content":     None,
            "tool_call":   None,
            "tool_result": avail_result,   # returns a conflict
        },
        {
            "turn": 4,
            "role": "assistant",
            # ★ Correct infeasible-task response: surface the constraint,
            #   do NOT attempt to create the event.
            "content": (
                "I wasn't able to book that slot — David Kim already has an event "
                "from 2:00 PM to 3:00 PM on July 11th, so there's a conflict.\n\n"
                "Would you like me to:\n"
                "  (a) Find the next available 1-hour slot for you and David that day, or\n"
                "  (b) Suggest a different day entirely?"
            ),
            "tool_call":   None,
            "tool_result": None,
            "annotation": {
                "infeasible": True,
                "reason":     "TIME_SLOT_UNAVAILABLE",
                "correct_behavior": (
                    "Assistant must NOT invoke calendar_create_event when a conflict "
                    "is detected. It must inform the user and offer alternatives."
                ),
            },
        },
    ]


def build_scenario_4() -> list[dict[str, Any]]:
    """
    Scenario 4 – Infeasible Task (Edge Case): Unresolvable email recipient.

    The user refers to a contact by name. The assistant correctly attempts to
    resolve the contact via `email_resolve_contact` before calling `email_send`.
    When resolution fails (contact not found), the assistant halts and reports
    the failure — it does NOT fabricate an email address.
    """
    resolve_args   = {"name": "Alex Turner"}
    resolve_result = execute_tool("email_resolve_contact", resolve_args)  # will fail

    return [
        {
            "turn": 1,
            "role": "user",
            "content": (
                "Send an email to Alex Turner with subject 'Project Update' and "
                "body: 'Hi Alex, just a quick heads-up that the report is ready.'"
            ),
            "tool_call":   None,
            "tool_result": None,
        },
        {
            "turn": 2,
            "role": "assistant",
            "content": "I'll look up Alex Turner's email address in your contacts first.",
            "tool_call": {
                "name":      "email_resolve_contact",
                "arguments": resolve_args,
            },
            "tool_result": None,
        },
        {
            "turn": 3,
            "role": "tool",
            "content":     None,
            "tool_call":   None,
            "tool_result": resolve_result,   # CONTACT_NOT_FOUND error
        },
        {
            "turn": 4,
            "role": "assistant",
            # ★ Correct behavior: halt, do NOT call email_send with a guessed address.
            "content": (
                "I couldn't find 'Alex Turner' in your contacts directory, so I "
                "can't send the email without risking it going to the wrong person.\n\n"
                "Could you provide Alex's email address directly, or clarify the "
                "full name so I can try again?"
            ),
            "tool_call":   None,
            "tool_result": None,
            "annotation": {
                "infeasible": True,
                "reason":     "CONTACT_NOT_FOUND",
                "correct_behavior": (
                    "Assistant must NOT call email_send with a fabricated or guessed "
                    "address. It must surface the contact resolution failure and ask "
                    "the user for clarification."
                ),
            },
        },
    ]


# ---------------------------------------------------------------------------
# 4. RUNNER  – build all scenarios and pretty-print as JSON
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: builds all scenarios and emits a single JSON structure."""

    all_scenarios: dict[str, Any] = {
        "tool_definitions": TOOL_DEFINITIONS,
        "scenarios": [
            {
                "id":          "scenario_1",
                "label":       "Happy Path – Direct Calendar Booking",
                "description": (
                    "Single-shot tool invocation. All required parameters are present "
                    "in the user's first message; no clarification needed."
                ),
                "turns": build_scenario_1(),
            },
            {
                "id":          "scenario_2",
                "label":       "Multi-Step – Availability Check then Book",
                "description": (
                    "Two sequential tool calls where the output of turn 3 "
                    "(availability data) directly constrains the arguments of turn 6 "
                    "(event creation). Tests multi-turn state tracking."
                ),
                "turns": build_scenario_2(),
            },
            {
                "id":          "scenario_3",
                "label":       "Edge Case – Infeasible: Conflicting Calendar Slot",
                "description": (
                    "Availability check returns a busy interval. The assistant must "
                    "recognise the constraint violation and NOT proceed with booking. "
                    "Correct behavior: surface the conflict, propose alternatives."
                ),
                "turns": build_scenario_3(),
            },
            {
                "id":          "scenario_4",
                "label":       "Edge Case – Infeasible: Unresolvable Email Recipient",
                "description": (
                    "Contact resolution fails (CONTACT_NOT_FOUND). The assistant must "
                    "NOT fabricate an email address. Correct behavior: halt the email "
                    "workflow and request clarification from the user."
                ),
                "turns": build_scenario_4(),
            },
        ],
    }

    print(json.dumps(all_scenarios, indent=2))


if __name__ == "__main__":
    main()
