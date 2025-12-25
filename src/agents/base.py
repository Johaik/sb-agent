from typing import List, Dict, Any, Optional
from ..llm.base import LLMProvider
from ..tools.base import Tool
from ..db.database import SessionLocal
from ..db.models import AgentLog

class Agent:
    def __init__(self, name: str, instructions: str, llm: LLMProvider, tools: List[Tool] = []):
        self.name = name
        self.instructions = instructions
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        # Deprecated: use log_event internally
        self.history.append({"role": role, "content": content})

    def log_event(self, role: str, content: str, tool_calls: list = None, invocation_state: dict = None):
        # Store in memory
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.history.append(msg)
        
        # Persist to DB if job_id is present
        if invocation_state and "job_id" in invocation_state:
            try:
                db = SessionLocal()
                log = AgentLog(
                    job_id=invocation_state["job_id"],
                    agent_name=self.name,
                    role=role,
                    content=content,
                    tool_calls=tool_calls
                )
                db.add(log)
                db.commit()
                db.close()
            except Exception as e:
                print(f"Failed to log event: {e}")

    def run(self, user_input: str, invocation_state: Dict[str, Any] = None) -> str:
        self.log_event("user", user_input, invocation_state=invocation_state)
        
        # Prepare messages: Instructions as system prompt or first user message?
        # Bedrock/Claude prefers system prompt separate, but our generic interface uses list of messages.
        # We'll prepend instructions to the history for the generation call, or use system role if provider supports it.
        # For simplicity, let's prepend as a user message or system message.
        # Note: 'system' role is supported by Claude 3 and recent OpenAI models.
        
        messages_to_send = [{"role": "system", "content": self.instructions}] + self.history

        # Loop for tool execution
        max_turns = 5
        current_turn = 0
        
        while current_turn < max_turns:
            current_turn += 1
            
            tool_definitions = [t.to_dict() for t in self.tools.values()]
            response = self.llm.generate(messages_to_send, tools=tool_definitions)
            
            content = response.get("content", "")
            tool_calls = response.get("tool_calls")
            
            if content:
                # self.add_message("assistant", content) # Don't add yet, waiting to see if tool calls exist
                pass

            # Update history with assistant response
            # Note: We need to be careful with how we construct the history for the *next* call.
            # The LLMProvider returns generic content/tool_calls. We need to format it back into the message history
            # consistent with the provider's expectation?
            # actually, our LLMProvider takes generic [{"role":..., "content":...}].
            # We need to standardize how tool calls are represented in history.
            
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                # In generic format, we might store tool_calls in metadata or content.
                # Standard OpenAI/Claude usage: assistant message has tool_calls.
                # We'll attach it.
                assistant_msg["tool_calls"] = tool_calls
            
            self.log_event("assistant", content, tool_calls=tool_calls, invocation_state=invocation_state)
            messages_to_send.append(assistant_msg)

            if not tool_calls:
                return content

            # Execute tools
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_input = tc["input"]
                tool_id = tc["id"]
                
                if tool_name in self.tools:
                    try:
                        print(f"[{self.name}] Executing tool {tool_name} with input {tool_input}")
                        result = self.tools[tool_name].run(invocation_state=invocation_state, **tool_input)
                        tool_output = str(result)
                    except Exception as e:
                        tool_output = f"Error executing tool: {str(e)}"
                else:
                    tool_output = f"Tool {tool_name} not found."
                
                # Add tool result to history
                # Generic format: role='tool' (OpenAI) or 'user' (Anthropic sometimes, but recently 'tool_result').
                # Let's use a 'tool' role and let the Provider handle adaptation if needed.
                # Actually, our providers currently don't handle adaptation of history *back* to the model specific format deeply.
                # 'user' is safest for generic, but 'tool' is semantic.
                # Let's use 'tool' and assume providers might need an update if they strictly reject it.
                # Bedrock Claude expects role='user' for tool results usually, or specific blocks.
                # OpenAI expects role='tool'.
                
                # To make this robust, let's keep it simple: generic 'tool' role.
                # We will need to update LLMProviders to handle 'tool' role in history conversion.
                
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": tool_output
                }
                self.log_event("tool", tool_output, invocation_state=invocation_state)
                # Note: log_event appends to self.history
                # But we constructed tool_msg manually above to match what we would append.
                # Actually log_event appends {"role":..., "content":...}, but tool_msg has extra fields.
                # Let's fix log_event to handle tool fields or just append manually here for memory, but use log_event for DB.
                # To keep it clean, let's just log to DB here separately if we want exact fields,
                # or better: rely on log_event for both but we need to pass extra fields to log_event if we want them in history.
                
                # Simplified approach: Use log_event for DB persistence mostly.
                # Re-aligning history:
                self.history[-1]["tool_call_id"] = tool_id
                self.history[-1]["name"] = tool_name
                
                messages_to_send.append(tool_msg)
                
        return "Max turns reached."

