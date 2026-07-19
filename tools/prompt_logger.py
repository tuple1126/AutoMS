import os
import sys
from datetime import datetime
from typing import List, Dict, Optional
from io import StringIO

class PromptLogger:
    def __init__(self, log_dir="logs/prompts"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create a new log file for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"prompt_log_{timestamp}.txt")
        
        # Terminal-output tracking.
        self._last_terminal_position = 0  # Output position at the previous log entry.
        self._terminal_buffer_ref = None  # Reference to the externally supplied buffer.
        
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"Prompt Log Session Started: {timestamp}\n")
            f.write("="*80 + "\n\n")

    def set_terminal_buffer(self, buffer_dict: dict):
        """Set the reference to the terminal-output buffer."""
        self._terminal_buffer_ref = buffer_dict
        self._last_terminal_position = 0

    def _get_new_terminal_output(self) -> str:
        """Return terminal output added since the previous log entry."""
        if self._terminal_buffer_ref is None:
            return ""
        
        # Read the current content from StringIO when available.
        try:
            # Check for a _buf_out reference that exposes StringIO directly.
            if hasattr(self._terminal_buffer_ref, 'get') and '_buf_out' in self._terminal_buffer_ref:
                buf = self._terminal_buffer_ref['_buf_out']
                current_text = buf.getvalue()
            elif 'text' in self._terminal_buffer_ref:
                current_text = self._terminal_buffer_ref.get('text', '')
            else:
                return ""
            
            # Extract only the newly added output.
            new_output = current_text[self._last_terminal_position:]
            self._last_terminal_position = len(current_text)
            return new_output
        except Exception as e:
            return f"[Error getting terminal output: {e}]"

    def log_prompt(self, agent_name: str, prompt_content: str):
        """Log the prompt seen by an agent."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        separator = "="*80
        
        log_entry = f"""
{separator}
TIMESTAMP: {timestamp}
AGENT: {agent_name}
{separator}
PROMPT CONTENT:
{prompt_content}
{separator}

"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error logging prompt: {e}")

    def log_full_agent_execution(
        self,
        agent_name: str,
        agent_role: str,
        agent_instructions: str,
        context_summary: str,
        manager_instruction: str,
        previous_output: str,
        augmented_query: str,
        history: List[Dict],
        response: str,
        terminal_output: str = None
    ):
        """
        Record the complete prompt visible to an agent, including:
        - System message (agent name, instructions, and role)
        - Conversation history
        - Complete user query (``augmented_query``)
        - Terminal output
        - Response
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        separator = "=" * 80
        sub_separator = "-" * 60
        
        log_entry = []
        log_entry.append(f"\n{separator}")
        log_entry.append(f"TIMESTAMP: {timestamp}")
        log_entry.append(f"AGENT: {agent_name}")
        log_entry.append(separator)
        
        # ========== 1. System message ==========
        log_entry.append("\n" + "="*30 + " SYSTEM MESSAGE " + "="*30)
        log_entry.append(f"## Agent name: {agent_name}")
        log_entry.append("## Agent instructions:")
        log_entry.append(f"{agent_instructions}")
        log_entry.append(f"## Role: {agent_role}")
        log_entry.append("Work through the user request step by step. Use relevant supplemental information and tools until you have the information required to answer the request.")
        log_entry.append(f"Date: {datetime.now().strftime('%Y-%m-%d')} Current time: {timestamp}")
        
        # ========== 2. Conversation history ==========
        log_entry.append("\n" + "="*30 + " HISTORY (last 5) " + "="*30)
        if history:
            for i, msg in enumerate(history[-5:]):  # Show only the five most recent entries.
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                # Truncate excessively long content.
                if len(content) > 800:
                    content = content[:800] + "\n... [truncated, total length: " + str(len(msg.get('content', ''))) + "]"
                log_entry.append(f"[{i+1}] {role}:")
                log_entry.append(content)
                log_entry.append(sub_separator)
        else:
            log_entry.append("[No history]")
        
        # ========== 3. Complete user query (augmented_query) ==========
        log_entry.append("\n" + "="*30 + " USER QUERY (augmented_query) " + "="*30)
        log_entry.append("Complete user query received by the agent:")
        log_entry.append(sub_separator)
        log_entry.append(augmented_query)
        log_entry.append(sub_separator)
        
        # ========== 4. Terminal output ==========
        # Read new terminal output.
        new_terminal = terminal_output or ""
        log_entry.append("\n" + "="*30 + " TERMINAL OUTPUT " + "="*30)
        if new_terminal and new_terminal.strip():
            log_entry.append(new_terminal.strip())
        else:
            log_entry.append("[No new terminal output]")
        
        # ========== 5. Agent response ==========
        log_entry.append("\n" + "="*30 + " AGENT RESPONSE " + "="*30)
        log_entry.append(response)
        
        log_entry.append(f"\n{separator}\n\n")
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write("\n".join(log_entry))
        except Exception as e:
            print(f"Error logging full execution: {e}")

    def log_execution(self, agent_name: str, context: str, instruction: str, response: str = None):
        """Log execution details: context + instruction -> response (legacy-compatible API)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        separator = "="*50
        
        log_entry = [
            f"\n{separator}",
            f"TIMESTAMP: {timestamp}",
            f"AGENT: {agent_name}",
            f"{separator}",
            "\n[BASIS / CONTEXT] (information used by the agent):",
            str(context).strip(),
            "\n[INSTRUCTION] (current task):",
            str(instruction).strip()
        ]
        
        if response:
            log_entry.extend([
                "\n[RESULT] (generated result):",
                str(response).strip()
            ])
            
        log_entry.append(f"{separator}\n\n")
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write("\n".join(log_entry))
        except Exception as e:
            print(f"Error logging execution: {e}")

# Global instance
_prompt_logger = None

def get_prompt_logger():
    global _prompt_logger
    if _prompt_logger is None:
        _prompt_logger = PromptLogger()
    return _prompt_logger
