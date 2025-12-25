import json
import structlog
from celery import chord, chain
from .celery_app import celery_app
from ..llm.factory import get_llm_provider
from ..agents.specialized import EnricherAgent, PlannerAgent, ResearcherAgent, ReporterAgent, CriticAgent
from ..db.database import SessionLocal
from ..db.models import ResearchReport, ResearchTask
from ..db.vector import save_chunks

logger = structlog.get_logger()

def clean_json_string(s: str) -> str:
    # Remove markdown code blocks if present
    s = s.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

@celery_app.task
def enrich_idea(idea: str, job_id: str):
    logger.info("enrich_idea_started", job_id=job_id)
    try:
        llm = get_llm_provider("bedrock")
        agent = EnricherAgent(llm)
        description = agent.run(idea, invocation_state={"job_id": job_id})
        
        # Update DB status
        db = SessionLocal()
        try:
            report = db.query(ResearchReport).filter(ResearchReport.id == job_id).first()
            if report:
                report.description = description
                report.status = "processing"
                db.commit()
        except Exception as e:
            logger.error("enrich_idea_db_update_failed", job_id=job_id, error=str(e))
        finally:
            db.close()
            
        logger.info("enrich_idea_completed", job_id=job_id)
        return description
    except Exception as e:
        logger.error("enrich_idea_failed", job_id=job_id, error=str(e))
        # We might want to fail the job here, but let's let celery handle retry/failure for now
        raise e

@celery_app.task
def plan_research(description: str, job_id: str):
    logger.info("plan_research_started", job_id=job_id)
    llm = get_llm_provider("bedrock")
    agent = PlannerAgent(llm)
    tasks_json = agent.run(description, invocation_state={"job_id": job_id})
    
    db = SessionLocal()
    try:
        tasks = json.loads(clean_json_string(tasks_json))
        if not isinstance(tasks, list):
            tasks = [description]
        
        # Create Task Entries in DB
        for t_title in tasks:
            new_task = ResearchTask(
                job_id=job_id,
                title=t_title,
                status="PENDING"
            )
            db.add(new_task)
        db.commit()
        
        logger.info("plan_research_completed", job_id=job_id, task_count=len(tasks))
        
        # Trigger Supervisor Loop
        supervisor_loop.delay(job_id)
        
    except Exception as e:
        logger.error("plan_research_failed", job_id=job_id, error=str(e))
        # Fallback: create one generic task
        new_task = ResearchTask(
            job_id=job_id,
            title=description,
            status="PENDING"
        )
        db.add(new_task)
        db.commit()
        supervisor_loop.delay(job_id)
        
    finally:
        db.close()

@celery_app.task
def perform_research_task(task_id: str):
    db = SessionLocal()
    task = None
    try:
        task = db.query(ResearchTask).filter(ResearchTask.id == task_id).first()
        if not task:
            logger.warning("perform_research_task_not_found", task_id=task_id)
            return
        
        logger.info("perform_research_task_started", task_id=task_id, job_id=str(task.job_id))
        
        llm = get_llm_provider("bedrock")
        agent = ResearcherAgent(llm)
        
        # Pass feedback if it was rejected previously
        result = agent.run_with_feedback(task.title, feedback=task.feedback, invocation_state={"job_id": str(task.job_id)})
        
        task.result = result
        task.status = "REVIEW"
        db.commit()
        
        logger.info("perform_research_task_completed", task_id=task_id)
        
        # Trigger Supervisor check
        supervisor_loop.delay(str(task.job_id))
        
    except Exception as e:
        logger.error("perform_research_task_failed", task_id=task_id, error=str(e))
        if task:
            task.status = "REJECTED"
            task.feedback = f"System Error: {str(e)}"
            db.commit()
            # Trigger supervisor to pick it up again (retry)
            supervisor_loop.delay(str(task.job_id))
    finally:
        db.close()

@celery_app.task
def review_task(task_id: str):
    db = SessionLocal()
    task = None
    try:
        task = db.query(ResearchTask).filter(ResearchTask.id == task_id).first()
        if not task:
            return
        
        logger.info("review_task_started", task_id=task_id, job_id=str(task.job_id))
        
        llm = get_llm_provider("bedrock")
        critic = CriticAgent(llm)
        
        critic_input = f"Task: {task.title}\n\nResult: {task.result}"
        response_json = critic.run(critic_input, invocation_state={"job_id": str(task.job_id)})
        
        try:
            review = json.loads(clean_json_string(response_json))
            approved = review.get("approved", False)
            feedback = review.get("feedback", "")
            
            if approved:
                task.status = "APPROVED"
                logger.info("task_approved", task_id=task_id)
            else:
                task.status = "REJECTED"
                task.feedback = feedback
                logger.info("task_rejected", task_id=task_id, feedback=feedback)
                
            db.commit()
            supervisor_loop.delay(str(task.job_id))
            
        except Exception as e:
            logger.error("review_task_parsing_failed", task_id=task_id, error=str(e))
            task.status = "REJECTED"
            task.feedback = f"Critic Error: {str(e)}"
            db.commit()
            supervisor_loop.delay(str(task.job_id))
            
    except Exception as e:
        logger.error("review_task_failed", task_id=task_id, error=str(e))
        if task:
            task.status = "REJECTED" 
            task.feedback = f"System Error in Review: {str(e)}"
            db.commit()
            supervisor_loop.delay(str(task.job_id))
    finally:
        db.close()

@celery_app.task
def aggregate_report(job_id: str):
    logger.info("aggregate_report_started", job_id=job_id)
    db = SessionLocal()
    try:
        # Get all approved tasks
        tasks = db.query(ResearchTask).filter(
            ResearchTask.job_id == job_id,
            ResearchTask.status == "APPROVED"
        ).all()
        
        results = [{"task": t.title, "result": t.result} for t in tasks]
        
        llm = get_llm_provider("bedrock")
        
        # 1. Embed and Save Chunks (Optional: done here or during research?)
        # Let's do it here to ensure we only index approved content.
        chunks_data = []
        for item in results:
            text = f"Task: {item['task']}\nResult: {item['result']}"
            try:
                embedding = llm.get_embedding(text)
                chunks_data.append({"content": text, "embedding": embedding})
            except Exception as e:
                logger.warning("embedding_failed", error=str(e))
        
        if chunks_data:
            save_chunks(db, job_id, chunks_data)
        
        # 2. Generate Final Report
        reporter = ReporterAgent(llm)
        all_content = "\n\n".join([r['result'] for r in results])
        final_json_str = reporter.run(all_content, invocation_state={"job_id": job_id})
        
        try:
            final_report = json.loads(clean_json_string(final_json_str))
        except:
            final_report = {"raw_output": final_json_str}
            
        # Update DB
        report = db.query(ResearchReport).filter(ResearchReport.id == job_id).first()
        if report:
            report.report = final_report
            report.status = "completed"
            db.commit()
            logger.info("job_completed", job_id=job_id)
            
    except Exception as e:
        logger.error("aggregate_report_failed", job_id=job_id, error=str(e))
    finally:
        db.close()

@celery_app.task
def supervisor_loop(job_id: str):
    db = SessionLocal()
    try:
        tasks = db.query(ResearchTask).filter(ResearchTask.job_id == job_id).all()
        
        all_approved = True
        
        for task in tasks:
            if task.status == "PENDING" or task.status == "REJECTED":
                all_approved = False
                task_id = str(task.id)
                task.status = "RESEARCHING"
                db.commit()
                perform_research_task.delay(task_id)
                
            elif task.status == "REVIEW":
                all_approved = False
                task_id = str(task.id)
                task.status = "REVIEWING"
                db.commit()
                review_task.delay(task_id)
            
            elif task.status != "APPROVED":
                # RESEARCHING, REVIEWING, etc.
                all_approved = False
        
        if all_approved and tasks:
            report = db.query(ResearchReport).filter(ResearchReport.id == job_id).first()
            if report and report.status != "completed" and report.status != "generating":
                report.status = "generating"
                db.commit()
                aggregate_report.delay(job_id)
                
    except Exception as e:
        logger.error("supervisor_loop_failed", job_id=job_id, error=str(e))
    finally:
        db.close()

def start_research_chain(idea: str, job_id: str):
    # This is called by the API
    # 1. Enrich
    # 2. Plan (which then triggers supervisor loop)
    logger.info("starting_research_chain", job_id=job_id)
    chain(
        enrich_idea.s(idea, job_id),
        plan_research.s(job_id)
    ).apply_async()
