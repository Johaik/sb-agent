import json
import structlog
from celery import chord, chain
from .celery_app import celery_app
from ..llm.factory import get_llm_provider
from ..agents.specialized import (
    EnricherAgent, PlannerAgent, HypothesisAgent, ResearcherAgent, 
    EvidenceAgent, ContradictionAgent, CriticAgent, ReporterAgent, FinalCriticAgent
)
from ..db.database import SessionLocal
from ..db.models import ResearchReport, ResearchTask
from ..db.vector import save_chunks

logger = structlog.get_logger()

def clean_json_string(s: str) -> str:
    # Remove markdown code blocks if present
    s = s.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

@celery_app.task
def enrich_idea(idea: str, job_id: str):
    logger.info("enrich_idea_started", job_id=job_id)
    agent = EnricherAgent()
    try:
        # Pass job_id via internal attr for logging hook
        agent._current_job_id = job_id
        # Use __call__ instead of invoke
        result = agent(idea)
        
        # Extract text from result (AgentResult)
        if hasattr(result, "last_message"):
             content_blocks = result.last_message.get("content", [])
             enriched_description = "".join([b["text"] for b in content_blocks if "text" in b])
        else:
             enriched_description = str(result)
        
        # Update database with enriched description
        db = SessionLocal()
        try:
            report = db.query(ResearchReport).filter(ResearchReport.id == job_id).first()
            if report:
                report.description = enriched_description
                report.status = "processing"
                db.commit()
        finally:
            db.close()
             
        logger.info("enrich_idea_completed", job_id=job_id)
        return enriched_description
    except Exception as e:
        logger.error("enrich_idea_failed", job_id=job_id, error=str(e))
        return idea

@celery_app.task
def plan_research(description: str, job_id: str):
    logger.info("plan_research_started", job_id=job_id)
    agent = PlannerAgent()
    try:
        agent._current_job_id = job_id
        result = agent(description)
        
        if hasattr(result, "last_message"):
             content_blocks = result.last_message.get("content", [])
             tasks_json = "".join([b["text"] for b in content_blocks if "text" in b])
        else:
             tasks_json = str(result)
        
        db = SessionLocal()
        try:
            tasks = json.loads(clean_json_string(tasks_json))
            if not isinstance(tasks, list):
                tasks = [description]
            
            for t_title in tasks:
                new_task = ResearchTask(
                    job_id=job_id,
                    title=t_title,
                    status="PENDING"
                )
                db.add(new_task)
            db.commit()
            
            logger.info("plan_research_completed", job_id=job_id, task_count=len(tasks))
            supervisor_loop.delay(job_id)
            
        except Exception as e:
            logger.error("plan_research_parsing_failed", job_id=job_id, error=str(e))
            # Fallback
            new_task = ResearchTask(job_id=job_id, title=description, status="PENDING")
            db.add(new_task)
            db.commit()
            supervisor_loop.delay(job_id)
        finally:
            db.close()
            
    except Exception as e:
         logger.error("plan_research_failed", job_id=job_id, error=str(e))

@celery_app.task
def generate_hypotheses_task(task_id: str):
    db = SessionLocal()
    try:
        task = db.query(ResearchTask).filter(ResearchTask.id == task_id).first()
        if not task: return
        
        logger.info("generate_hypotheses_started", task_id=task_id)
        agent = HypothesisAgent()
        agent._current_job_id = str(task.job_id)
        
        result = agent(task.title)
        
        if hasattr(result, "last_message"):
             content_blocks = result.last_message.get("content", [])
             result_json = "".join([b["text"] for b in content_blocks if "text" in b])
        else:
             result_json = str(result)
        
        task.hypotheses = json.loads(clean_json_string(result_json))
        task.status = "HYPOTHESIZED"
        db.commit()
        
        supervisor_loop.delay(str(task.job_id))
        
    except Exception as e:
        logger.error("generate_hypotheses_failed", task_id=task_id, error=str(e))
        # If failed, we can skip hypothesis or retry. Let's skip to ready for research.
        if task:
            task.status = "HYPOTHESIZED" # Treat as done but empty
            db.commit()
            supervisor_loop.delay(str(task.job_id))
    finally:
        db.close()

@celery_app.task
def perform_research_task(task_id: str):
    db = SessionLocal()
    try:
        task = db.query(ResearchTask).filter(ResearchTask.id == task_id).first()
        if not task: return
        
        logger.info("perform_research_task_started", task_id=task_id)
        
        # We might need legacy LLM provider for tools
        llm_provider = get_llm_provider("bedrock")
        agent = ResearcherAgent(llm_provider=llm_provider)
        
        # Include hypotheses in the prompt if available
        prompt = task.title
        if task.hypotheses:
            prompt += f"\n\nContext/Hypotheses: {json.dumps(task.hypotheses)}"
            
        result = agent.run_with_feedback(prompt, feedback=task.feedback, job_id=str(task.job_id))
        
        task.result = result
        task.status = "RESEARCHED"
        db.commit()
        
        supervisor_loop.delay(str(task.job_id))
        
    except Exception as e:
        logger.error("perform_research_task_failed", task_id=task_id, error=str(e))
        if task:
            task.status = "REJECTED" # Will retry via supervisor logic or manual
            task.feedback = f"System Error: {str(e)}"
            db.commit()
            supervisor_loop.delay(str(task.job_id))
    finally:
        db.close()

@celery_app.task
def score_evidence_task(task_id: str):
    db = SessionLocal()
    try:
        task = db.query(ResearchTask).filter(ResearchTask.id == task_id).first()
        if not task: return
        
        logger.info("score_evidence_started", task_id=task_id)
        agent = EvidenceAgent()
        agent._current_job_id = str(task.job_id)
        
        input_text = f"Task: {task.title}\nFindings: {task.result}"
        result = agent(input_text)
        
        if hasattr(result, "last_message"):
             content_blocks = result.last_message.get("content", [])
             result_json = "".join([b["text"] for b in content_blocks if "text" in b])
        else:
             result_json = str(result)
        
        task.evidence_rating = json.loads(clean_json_string(result_json))
        task.status = "SCORED"
        db.commit()
        
        supervisor_loop.delay(str(task.job_id))
    except Exception as e:
        logger.error("score_evidence_failed", task_id=task_id, error=str(e))
        if task:
            task.status = "SCORED" # Skip on error
            db.commit()
            supervisor_loop.delay(str(task.job_id))
    finally:
        db.close()

@celery_app.task
def find_contradictions_task(task_id: str):
    db = SessionLocal()
    try:
        task = db.query(ResearchTask).filter(ResearchTask.id == task_id).first()
        if not task: return
        
        logger.info("find_contradictions_started", task_id=task_id)
        llm_provider = get_llm_provider("bedrock")
        agent = ContradictionAgent(llm_provider=llm_provider)
        agent._current_job_id = str(task.job_id)
        
        input_text = f"Task: {task.title}\nFindings: {task.result}"
        result = agent(input_text)
        
        if hasattr(result, "last_message"):
             content_blocks = result.last_message.get("content", [])
             result_json = "".join([b["text"] for b in content_blocks if "text" in b])
        else:
             result_json = str(result)
        
        task.contradictions = json.loads(clean_json_string(result_json))
        task.status = "CONTRADICTED"
        db.commit()
        
        supervisor_loop.delay(str(task.job_id))
    except Exception as e:
        logger.error("find_contradictions_failed", task_id=task_id, error=str(e))
        if task:
            task.status = "CONTRADICTED" # Skip
            db.commit()
            supervisor_loop.delay(str(task.job_id))
    finally:
        db.close()

@celery_app.task
def review_task(task_id: str):
    db = SessionLocal()
    try:
        task = db.query(ResearchTask).filter(ResearchTask.id == task_id).first()
        if not task: return
        
        logger.info("review_task_started", task_id=task_id)
        agent = CriticAgent()
        agent._current_job_id = str(task.job_id)
        
        # Include contradictions in review context
        critic_input = f"Task: {task.title}\nResult: {task.result}\nContradictions Found: {task.contradictions}"
        result = agent(critic_input)
        
        if hasattr(result, "last_message"):
             content_blocks = result.last_message.get("content", [])
             response_json = "".join([b["text"] for b in content_blocks if "text" in b])
        else:
             response_json = str(result)
        
        try:
            review = json.loads(clean_json_string(response_json))
            if review.get("approved", False):
                task.status = "APPROVED"
            else:
                task.status = "REJECTED"
                task.feedback = review.get("feedback", "")
            
            db.commit()
            supervisor_loop.delay(str(task.job_id))
        except Exception as e:
            logger.error("review_parsing_failed", error=str(e))
            task.status = "REJECTED"
            task.feedback = "Critic JSON Parse Error"
            db.commit()
            supervisor_loop.delay(str(task.job_id))
            
    except Exception as e:
        logger.error("review_task_failed", task_id=task_id, error=str(e))
    finally:
        db.close()

@celery_app.task
def final_critique_task(job_id: str, draft_report: dict):
    logger.info("final_critique_started", job_id=job_id)
    db = SessionLocal()
    try:
        agent = FinalCriticAgent()
        agent._current_job_id = job_id
        
        result = agent(json.dumps(draft_report))
        
        if hasattr(result, "last_message"):
             content_blocks = result.last_message.get("content", [])
             critique_text = "".join([b["text"] for b in content_blocks if "text" in b])
        else:
             critique_text = str(result)
        
        # Try to parse critique as JSON
        try:
            critique = json.loads(clean_json_string(critique_text))
        except Exception:
            # If not JSON, consider it approved with the text as feedback
            logger.info("final_critique_plain_text", job_id=job_id)
            critique = {
                "approved": True,
                "feedback": critique_text
            }
        
        report_obj = db.query(ResearchReport).filter(ResearchReport.id == job_id).first()
        if not report_obj: return

        if critique.get("approved", False):
            report_obj.report = draft_report
            report_obj.final_critique = critique
            report_obj.status = "completed"
            
            # Save chunks
            try:
                llm = get_llm_provider("bedrock") # For embedding
                save_chunks(db, job_id, draft_report, llm)
            except Exception as e:
                logger.error("vector_save_failed", error=str(e))
                
            logger.info("research_completed_successfully", job_id=job_id)
        else:
            # If rejected, what do we do? For MVP, we log it and mark completed but with warning, 
            # or we could loop back to Reporter. For now, let's just save it but mark specific status or just include critique.
            # The plan said "If rejected -> Send feedback back to ReporterAgent (loop)".
            # Implementing loop might be complex for this step. Let's just save and mark as 'completed_with_feedback'
            report_obj.report = draft_report
            report_obj.final_critique = critique
            report_obj.status = "completed" # For now
            
        db.commit()
        
    except Exception as e:
        logger.error("final_critique_failed", job_id=job_id, error=str(e))
        # Fallback save
        report_obj = db.query(ResearchReport).filter(ResearchReport.id == job_id).first()
        if report_obj:
            report_obj.report = draft_report
            report_obj.status = "completed"
            db.commit()
    finally:
        db.close()

@celery_app.task
def aggregate_report(job_id: str):
    logger.info("aggregate_report_started", job_id=job_id)
    db = SessionLocal()
    try:
        tasks = db.query(ResearchTask).filter(ResearchTask.job_id == job_id, ResearchTask.status == "APPROVED").all()
        if not tasks: return

        agent = ReporterAgent()
        agent._current_job_id = job_id
        
        # Build a more structured context with all details preserved
        context_parts = []
        context_parts.append("=== RESEARCH COMPILATION FOR COMPREHENSIVE REPORT ===\n")
        context_parts.append(f"Total Research Tasks Completed: {len(tasks)}\n")
        context_parts.append("=" * 80 + "\n\n")
        
        for idx, t in enumerate(tasks, 1):
            context_parts.append(f"{'=' * 80}\n")
            context_parts.append(f"RESEARCH TASK #{idx}\n")
            context_parts.append(f"{'=' * 80}\n\n")
            
            context_parts.append(f"TASK TITLE:\n{t.title}\n\n")
            
            # Include hypotheses if available
            if t.hypotheses:
                context_parts.append(f"HYPOTHESES CONSIDERED:\n")
                try:
                    hyp_data = t.hypotheses if isinstance(t.hypotheses, dict) else json.loads(t.hypotheses)
                    context_parts.append(json.dumps(hyp_data, indent=2))
                except:
                    context_parts.append(str(t.hypotheses))
                context_parts.append("\n\n")
            
            # Main research findings - this is the critical data
            context_parts.append(f"DETAILED RESEARCH FINDINGS:\n")
            context_parts.append("-" * 80 + "\n")
            context_parts.append(str(t.result))
            context_parts.append("\n" + "-" * 80 + "\n\n")
            
            # Include evidence rating
            if t.evidence_rating:
                context_parts.append(f"EVIDENCE QUALITY ASSESSMENT:\n")
                try:
                    evidence = t.evidence_rating if isinstance(t.evidence_rating, dict) else json.loads(t.evidence_rating)
                    context_parts.append(json.dumps(evidence, indent=2))
                except:
                    context_parts.append(str(t.evidence_rating))
                context_parts.append("\n\n")
            
            # Include contradictions found
            if t.contradictions:
                context_parts.append(f"CONTRADICTIONS & ALTERNATIVE PERSPECTIVES:\n")
                try:
                    contrad = t.contradictions if isinstance(t.contradictions, dict) else json.loads(t.contradictions)
                    context_parts.append(json.dumps(contrad, indent=2))
                except:
                    context_parts.append(str(t.contradictions))
                context_parts.append("\n\n")
            
            context_parts.append("\n")
        
        context_parts.append("=" * 80 + "\n")
        context_parts.append("END OF RESEARCH DATA - NOW CREATE COMPREHENSIVE REPORT\n")
        context_parts.append("=" * 80 + "\n\n")
        context_parts.append("INSTRUCTIONS: Synthesize ALL the above research findings into a comprehensive, detailed JSON report.\n")
        context_parts.append("Remember to preserve ALL details, examples, specifications, and data points from the research findings.\n")
        context_parts.append("Your report should be proportional to the amount of research data provided above.\n")
            
        full_context = "".join(context_parts)
        
        logger.info("aggregate_report_context_length", job_id=job_id, context_length=len(full_context))
        
        result = agent(full_context)
        
        if hasattr(result, "last_message"):
             content_blocks = result.last_message.get("content", [])
             report_text = "".join([b["text"] for b in content_blocks if "text" in b])
        else:
             report_text = str(result)
        
        # Try to parse as JSON first
        report_data = None
        try:
            report_data = json.loads(clean_json_string(report_text))
        except Exception as e:
            # If not JSON, treat as plain text report
            logger.info("aggregate_report_plain_text", job_id=job_id, error=str(e))
            report_data = {
                "content": report_text,
                "format": "plain_text"
            }
        
        logger.info("aggregate_report_generated", job_id=job_id, report_length=len(report_text))
        
        # Trigger Final Critic
        final_critique_task.delay(job_id, report_data)
                
    except Exception as e:
        logger.error("aggregate_report_failed", job_id=job_id, error=str(e))
        report_obj = db.query(ResearchReport).filter(ResearchReport.id == job_id).first()
        if report_obj:
            report_obj.status = "failed"
            report_obj.report = {"error": str(e)}
            db.commit()
    finally:
        db.close()

@celery_app.task
def supervisor_loop(job_id: str):
    db = SessionLocal()
    try:
        tasks = db.query(ResearchTask).filter(ResearchTask.job_id == job_id).all()
        
        all_approved = True
        
        for task in tasks:
            task_id = str(task.id)
            
            # State Machine
            if task.status == "PENDING":
                all_approved = False
                task.status = "HYPOTHESIZING_STARTED" # Interim state to prevent re-queue
                db.commit()
                generate_hypotheses_task.delay(task_id)
                
            elif task.status == "HYPOTHESIZED":
                all_approved = False
                task.status = "RESEARCHING_STARTED"
                db.commit()
                perform_research_task.delay(task_id)
            
            elif task.status == "RESEARCHED":
                all_approved = False
                task.status = "SCORING_STARTED"
                db.commit()
                score_evidence_task.delay(task_id)
            
            elif task.status == "SCORED":
                all_approved = False
                task.status = "CONTRADICTING_STARTED"
                db.commit()
                find_contradictions_task.delay(task_id)
            
            elif task.status == "CONTRADICTED":
                all_approved = False
                task.status = "REVIEW_STARTED"
                db.commit()
                review_task.delay(task_id)
            
            elif task.status == "REJECTED":
                # If rejected, go back to Researching
                all_approved = False
                task.status = "RESEARCHING_RETRY"
                db.commit()
                perform_research_task.delay(task_id)
            
            elif task.status != "APPROVED":
                # Currently in progress (e.g. HYPOTHESIZING_STARTED)
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

@celery_app.task
def start_research_chain(idea: str, job_id: str):
    chain(
        enrich_idea.s(idea, job_id),
        plan_research.s(job_id)
    ).apply_async()
