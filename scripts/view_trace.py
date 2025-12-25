import sys
import os
import argparse

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db.database import SessionLocal
from src.db.models import AgentLog, ResearchReport

def view_trace(job_id):
    db = SessionLocal()
    try:
        logs = db.query(AgentLog).filter(AgentLog.job_id == job_id).order_by(AgentLog.timestamp).all()
        
        if not logs:
            print(f"No logs found for job {job_id}")
            return

        print(f"\n=== TRACE FOR JOB {job_id} ===\n")
        for log in logs:
            timestamp = log.timestamp.strftime('%H:%M:%S')
            print(f"[{timestamp}] {log.agent_name} ({log.role})")
            
            if log.content:
                print(f"Content: {log.content}")
            
            if log.tool_calls:
                print(f"Tools Requested: {log.tool_calls}")
                
            print("-" * 60)
            
    except Exception as e:
        print(f"Error viewing trace: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View agent execution trace")
    parser.add_argument("job_id", help="The UUID of the research job")
    args = parser.parse_args()
    
    view_trace(args.job_id)

