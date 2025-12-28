import sys
import os

# Add the project root to the python path
sys.path.append(os.getcwd())

try:
    from src.worker import tasks
    print("Successfully imported tasks")
except Exception as e:
    print(f"Error importing tasks: {e}")
    import traceback
    traceback.print_exc()

