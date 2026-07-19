import os
from datetime import datetime
from tools.case_library import CaseLibraryManager

class SessionLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.case_manager = CaseLibraryManager()

    def save_terminal_log(self, content: str) -> str:
        """Save terminal output to a timestamped file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"terminal_log_{timestamp}.txt"
        filepath = os.path.join(self.log_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"\nTerminal output saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving terminal log: {e}")
            return ""

    def handle_session_end(self, user_requirement: str, terminal_output: str):
        """Handle end of session: save log and ask to save case."""
        # 1. Save raw terminal log
        self.save_terminal_log(terminal_output)

        # 2. Ask user to save to case library
        # print("\n" + "="*40)
        # print("Session Analysis & Archiving")
        # print("="*40)
        
        # The TreePlanTracker now handles session recording and case library functions.
        # The original CaseLibraryManager interaction is disabled to avoid redundancy.
        pass

