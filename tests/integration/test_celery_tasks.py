"""
Integration tests for Celery tasks.
Tests individual tasks with mocked agents.
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from uuid import uuid4


@pytest.fixture
def mock_agent_result():
    """Create a mock agent result."""
    result = MagicMock()
    result.last_message = {
        "content": [{"text": "Mocked response from agent"}]
    }
    return result


@pytest.fixture
def mock_agents(mock_agent_result):
    """Mock all agent classes."""
    with patch('src.worker.tasks.EnricherAgent') as enricher, \
         patch('src.worker.tasks.PlannerAgent') as planner, \
         patch('src.worker.tasks.HypothesisAgent') as hypothesis, \
         patch('src.worker.tasks.ResearcherAgent') as researcher, \
         patch('src.worker.tasks.EvidenceAgent') as evidence, \
         patch('src.worker.tasks.ContradictionAgent') as contradiction, \
         patch('src.worker.tasks.CriticAgent') as critic, \
         patch('src.worker.tasks.ReporterAgent') as reporter, \
         patch('src.worker.tasks.FinalCriticAgent') as final_critic:
        
        # Configure all agent mocks to return the mock result
        for agent_mock in [enricher, planner, hypothesis, researcher, 
                          evidence, contradiction, critic, reporter, final_critic]:
            agent_instance = MagicMock()
            agent_instance.return_value = mock_agent_result
            agent_mock.return_value = agent_instance
        
        yield {
            'enricher': enricher,
            'planner': planner,
            'hypothesis': hypothesis,
            'researcher': researcher,
            'evidence': evidence,
            'contradiction': contradiction,
            'critic': critic,
            'reporter': reporter,
            'final_critic': final_critic
        }


@pytest.mark.integration
class TestEnrichIdeaTask:
    """Tests for enrich_idea task."""
    
    def test_enrich_idea_updates_database(self, db_session, sample_research_report, mock_agents):
        """Test enrich_idea updates the report in database."""
        from src.worker.tasks import enrich_idea
        
        # Configure enricher to return specific text
        mock_result = MagicMock()
        mock_result.last_message = {
            "content": [{"text": "This is an enriched description of the research topic."}]
        }
        mock_agents['enricher'].return_value.return_value = mock_result
        
        job_id = str(sample_research_report.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_report
            mock_session.return_value = mock_db
            
            result = enrich_idea("Test idea", job_id)
        
        assert "enriched description" in result
    
    def test_enrich_idea_returns_original_on_error(self, db_session, mock_agents):
        """Test enrich_idea returns original idea on error."""
        from src.worker.tasks import enrich_idea
        
        mock_agents['enricher'].return_value.side_effect = Exception("Agent error")
        
        job_id = str(uuid4())
        result = enrich_idea("Original idea", job_id)
        
        assert result == "Original idea"


@pytest.mark.integration
class TestPlanResearchTask:
    """Tests for plan_research task."""
    
    def test_plan_research_creates_tasks(self, db_session, sample_research_report, mock_agents):
        """Test plan_research creates research tasks."""
        from src.worker.tasks import plan_research
        
        # Configure planner to return task list
        mock_result = MagicMock()
        mock_result.last_message = {
            "content": [{"text": '["Task 1", "Task 2", "Task 3"]'}]
        }
        mock_agents['planner'].return_value.return_value = mock_result
        
        job_id = str(sample_research_report.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.supervisor_loop') as mock_supervisor:
            
            mock_db = MagicMock()
            mock_session.return_value = mock_db
            
            plan_research("Test description", job_id)
            
            # Verify tasks were added
            assert mock_db.add.call_count >= 1
            mock_db.commit.assert_called()
            mock_supervisor.delay.assert_called_with(job_id)
    
    def test_plan_research_handles_invalid_json(self, db_session, sample_research_report, mock_agents):
        """Test plan_research handles invalid JSON response."""
        from src.worker.tasks import plan_research
        
        # Configure planner to return invalid JSON
        mock_result = MagicMock()
        mock_result.last_message = {
            "content": [{"text": "Not valid JSON"}]
        }
        mock_agents['planner'].return_value.return_value = mock_result
        
        job_id = str(sample_research_report.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.supervisor_loop') as mock_supervisor:
            
            mock_db = MagicMock()
            mock_session.return_value = mock_db
            
            # Should not raise, uses fallback
            plan_research("Test description", job_id)
            
            mock_supervisor.delay.assert_called()


@pytest.mark.integration
class TestGenerateHypothesesTask:
    """Tests for generate_hypotheses_task."""
    
    def test_generate_hypotheses_updates_task(self, db_session, sample_research_task, mock_agents):
        """Test hypotheses are saved to task."""
        from src.worker.tasks import generate_hypotheses_task
        
        # Configure hypothesis agent
        mock_result = MagicMock()
        mock_result.last_message = {
            "content": [{
                "text": '{"hypotheses": [{"statement": "Test", "confidence": "high", "reasoning": "Because"}]}'
            }]
        }
        mock_agents['hypothesis'].return_value.return_value = mock_result
        
        task_id = str(sample_research_task.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.supervisor_loop') as mock_supervisor:
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_task
            mock_session.return_value = mock_db
            
            generate_hypotheses_task(task_id)
            
            assert sample_research_task.status == "HYPOTHESIZED"
            mock_db.commit.assert_called()


@pytest.mark.integration
class TestPerformResearchTask:
    """Tests for perform_research_task."""
    
    def test_perform_research_saves_result(self, db_session, sample_research_task, mock_agents):
        """Test research result is saved to task."""
        from src.worker.tasks import perform_research_task
        
        # Configure researcher
        mock_agents['researcher'].return_value.run_with_feedback.return_value = "Detailed research findings"
        
        task_id = str(sample_research_task.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.supervisor_loop') as mock_supervisor, \
             patch('src.worker.tasks.get_llm_provider') as mock_llm:
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_task
            mock_session.return_value = mock_db
            
            perform_research_task(task_id)
            
            assert sample_research_task.result == "Detailed research findings"
            assert sample_research_task.status == "RESEARCHED"


@pytest.mark.integration
class TestScoreEvidenceTask:
    """Tests for score_evidence_task."""
    
    def test_score_evidence_saves_rating(self, db_session, sample_research_task, mock_agents):
        """Test evidence rating is saved."""
        from src.worker.tasks import score_evidence_task
        
        mock_result = MagicMock()
        mock_result.last_message = {
            "content": [{
                "text": '{"relevance_score": 8, "credibility_score": 7, "analysis": "Good", "weak_points": []}'
            }]
        }
        mock_agents['evidence'].return_value.return_value = mock_result
        
        sample_research_task.result = "Some research result"
        task_id = str(sample_research_task.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.supervisor_loop') as mock_supervisor:
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_task
            mock_session.return_value = mock_db
            
            score_evidence_task(task_id)
            
            assert sample_research_task.evidence_rating is not None
            assert sample_research_task.status == "SCORED"


@pytest.mark.integration
class TestFindContradictionsTask:
    """Tests for find_contradictions_task."""
    
    def test_find_contradictions_saves_result(self, db_session, sample_research_task, mock_agents):
        """Test contradictions are saved."""
        from src.worker.tasks import find_contradictions_task
        
        mock_result = MagicMock()
        mock_result.last_message = {
            "content": [{
                "text": '{"contradictions_found": false, "details": []}'
            }]
        }
        mock_agents['contradiction'].return_value.return_value = mock_result
        
        sample_research_task.result = "Research result"
        task_id = str(sample_research_task.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.supervisor_loop') as mock_supervisor, \
             patch('src.worker.tasks.get_llm_provider') as mock_llm:
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_task
            mock_session.return_value = mock_db
            
            find_contradictions_task(task_id)
            
            assert sample_research_task.status == "CONTRADICTED"


@pytest.mark.integration
class TestReviewTask:
    """Tests for review_task."""
    
    def test_review_task_approves(self, db_session, sample_research_task, mock_agents):
        """Test critic approves task."""
        from src.worker.tasks import review_task
        
        mock_result = MagicMock()
        mock_result.last_message = {
            "content": [{
                "text": '{"approved": true, "feedback": "Good research"}'
            }]
        }
        mock_agents['critic'].return_value.return_value = mock_result
        
        sample_research_task.result = "Research result"
        task_id = str(sample_research_task.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.supervisor_loop') as mock_supervisor:
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_task
            mock_session.return_value = mock_db
            
            review_task(task_id)
            
            assert sample_research_task.status == "APPROVED"
    
    def test_review_task_rejects(self, db_session, sample_research_task, mock_agents):
        """Test critic rejects task with feedback."""
        from src.worker.tasks import review_task
        
        mock_result = MagicMock()
        mock_result.last_message = {
            "content": [{
                "text": '{"approved": false, "feedback": "Needs more detail"}'
            }]
        }
        mock_agents['critic'].return_value.return_value = mock_result
        
        sample_research_task.result = "Research result"
        task_id = str(sample_research_task.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.supervisor_loop') as mock_supervisor:
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_task
            mock_session.return_value = mock_db
            
            review_task(task_id)
            
            assert sample_research_task.status == "REJECTED"
            assert sample_research_task.feedback == "Needs more detail"


@pytest.mark.integration
class TestAggregateReportTask:
    """Tests for aggregate_report task."""
    
    def test_aggregate_report_generates_report(self, db_session, sample_research_report, mock_agents):
        """Test report aggregation."""
        from src.worker.tasks import aggregate_report
        from src.db.models import ResearchTask
        
        # Create approved tasks
        tasks = []
        for i in range(2):
            task = ResearchTask(
                job_id=sample_research_report.id,
                title=f"Task {i}",
                status="APPROVED",
                result=f"Research result {i}"
            )
            tasks.append(task)
        
        mock_result = MagicMock()
        mock_result.last_message = {
            "content": [{
                "text": json.dumps({
                    "summary": "Test summary",
                    "key_findings": ["Finding 1"],
                    "details": {"Section 1": "Content"}
                })
            }]
        }
        mock_agents['reporter'].return_value.return_value = mock_result
        
        job_id = str(sample_research_report.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.final_critique_task') as mock_final:
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.all.return_value = tasks
            mock_session.return_value = mock_db
            
            aggregate_report(job_id)
            
            mock_final.delay.assert_called_once()


@pytest.mark.integration
class TestFinalCritiqueTask:
    """Tests for final_critique_task."""
    
    def test_final_critique_approves(self, db_session, sample_research_report, mock_agents):
        """Test final critic approves report."""
        from src.worker.tasks import final_critique_task
        
        mock_result = MagicMock()
        mock_result.last_message = {
            "content": [{
                "text": '{"approved": true, "critique": "Well done"}'
            }]
        }
        mock_agents['final_critic'].return_value.return_value = mock_result
        
        draft_report = {"summary": "Test", "key_findings": [], "details": {}}
        job_id = str(sample_research_report.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.get_llm_provider') as mock_llm, \
             patch('src.worker.tasks.save_chunks') as mock_save:
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_report
            mock_session.return_value = mock_db
            mock_llm.return_value = MagicMock()
            
            final_critique_task(job_id, draft_report)
            
            assert sample_research_report.status == "completed"
            assert sample_research_report.report == draft_report


@pytest.mark.integration
class TestSupervisorLoop:
    """Tests for supervisor_loop task."""
    
    def test_supervisor_triggers_next_stage(self, db_session, sample_research_report):
        """Test supervisor advances task through stages."""
        from src.worker.tasks import supervisor_loop
        from src.db.models import ResearchTask
        
        # Create a pending task
        task = ResearchTask(
            job_id=sample_research_report.id,
            title="Test task",
            status="PENDING"
        )
        
        job_id = str(sample_research_report.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.generate_hypotheses_task') as mock_hypothesis:
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.all.return_value = [task]
            mock_session.return_value = mock_db
            
            supervisor_loop(job_id)
            
            assert task.status == "HYPOTHESIZING_STARTED"
            mock_hypothesis.delay.assert_called()
    
    def test_supervisor_triggers_report_when_all_approved(self, db_session, sample_research_report):
        """Test supervisor triggers report when all tasks approved."""
        from src.worker.tasks import supervisor_loop
        from src.db.models import ResearchTask
        
        # All tasks approved
        tasks = [
            MagicMock(status="APPROVED", id=uuid4()),
            MagicMock(status="APPROVED", id=uuid4())
        ]
        
        job_id = str(sample_research_report.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.aggregate_report') as mock_report:
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.all.return_value = tasks
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_report
            mock_session.return_value = mock_db
            
            supervisor_loop(job_id)
            
            mock_report.delay.assert_called_with(job_id)


@pytest.mark.integration
class TestStartResearchChain:
    """Tests for start_research_chain task."""
    
    def test_start_chain_triggers_enrich_and_plan(self):
        """Test start_research_chain creates the initial chain."""
        from src.worker.tasks import start_research_chain
        
        with patch('src.worker.tasks.chain') as mock_chain:
            mock_chain_instance = MagicMock()
            mock_chain.return_value = mock_chain_instance
            
            start_research_chain("Test idea", "job-123")
            
            mock_chain.assert_called_once()
            mock_chain_instance.apply_async.assert_called_once()


@pytest.mark.integration
class TestTaskRetryAndErrorRecovery:
    """Tests for task retry logic and error recovery."""
    
    def test_rejected_task_triggers_retry(self, db_session, sample_research_report):
        """Test supervisor sends rejected tasks back to research phase."""
        from src.worker.tasks import supervisor_loop
        from src.db.models import ResearchTask
        
        # Create a rejected task
        task = MagicMock()
        task.id = uuid4()
        task.job_id = sample_research_report.id
        task.status = "REJECTED"
        task.feedback = "Need more details"
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.perform_research_task') as mock_research:
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.all.return_value = [task]
            mock_session.return_value = mock_db
            
            supervisor_loop(str(sample_research_report.id))
            
            # Status should change to RESEARCHING_RETRY
            assert task.status == "RESEARCHING_RETRY"
            mock_research.delay.assert_called_once()
    
    def test_enrich_idea_fallback_on_error(self, db_session, mock_agents):
        """Test enrich_idea returns original idea when agent fails."""
        from src.worker.tasks import enrich_idea
        
        mock_agents['enricher'].return_value.side_effect = RuntimeError("LLM Unavailable")
        
        job_id = str(uuid4())
        result = enrich_idea("Original research idea", job_id)
        
        # Should return the original idea as fallback
        assert result == "Original research idea"
    
    def test_generate_hypotheses_recovers_on_parse_error(self, db_session, sample_research_task, mock_agents):
        """Test hypothesis generation recovers when JSON parsing fails."""
        from src.worker.tasks import generate_hypotheses_task
        
        # Return invalid JSON
        mock_result = MagicMock()
        mock_result.last_message = {
            "content": [{"text": "Not valid JSON at all {{{"}]
        }
        mock_agents['hypothesis'].return_value.return_value = mock_result
        
        task_id = str(sample_research_task.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.supervisor_loop') as mock_supervisor:
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_task
            mock_session.return_value = mock_db
            
            # Should not raise
            generate_hypotheses_task(task_id)
            
            # Task should still progress (status changes to HYPOTHESIZED)
            mock_supervisor.delay.assert_called()
    
    def test_perform_research_marks_rejected_on_error(self, db_session, sample_research_task, mock_agents):
        """Test research task is marked REJECTED with feedback on error."""
        from src.worker.tasks import perform_research_task
        
        # Make researcher raise exception
        mock_agents['researcher'].return_value.run_with_feedback.side_effect = Exception("Network timeout")
        
        task_id = str(sample_research_task.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.supervisor_loop') as mock_supervisor, \
             patch('src.worker.tasks.get_llm_provider'):
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_task
            mock_session.return_value = mock_db
            
            perform_research_task(task_id)
            
            assert sample_research_task.status == "REJECTED"
            assert "Network timeout" in sample_research_task.feedback
            mock_supervisor.delay.assert_called()
    
    def test_review_task_handles_parsing_error(self, db_session, sample_research_task, mock_agents):
        """Test review task handles JSON parse errors gracefully."""
        from src.worker.tasks import review_task
        
        # Return non-JSON response
        mock_result = MagicMock()
        mock_result.last_message = {
            "content": [{"text": "This task looks good but I can't format JSON properly {unclosed"}]
        }
        mock_agents['critic'].return_value.return_value = mock_result
        
        sample_research_task.result = "Research findings"
        task_id = str(sample_research_task.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.supervisor_loop') as mock_supervisor:
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_task
            mock_session.return_value = mock_db
            
            review_task(task_id)
            
            # Should be rejected with parse error feedback
            assert sample_research_task.status == "REJECTED"
            assert "Parse Error" in sample_research_task.feedback
    
    def test_final_critique_fallback_saves_report(self, db_session, sample_research_report, mock_agents):
        """Test final_critique saves report even on error."""
        from src.worker.tasks import final_critique_task
        
        # Agent raises exception
        mock_agents['final_critic'].return_value.side_effect = Exception("Agent crashed")
        
        draft_report = {"summary": "Test report", "key_findings": [], "details": {}}
        job_id = str(sample_research_report.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_report
            mock_session.return_value = mock_db
            
            final_critique_task(job_id, draft_report)
            
            # Report should still be saved
            assert sample_research_report.report == draft_report
            assert sample_research_report.status == "completed"
    
    def test_aggregate_report_marks_failed_on_error(self, db_session, sample_research_report, mock_agents):
        """Test aggregate_report marks job as failed on unrecoverable error."""
        from src.worker.tasks import aggregate_report
        from src.db.models import ResearchTask
        
        # Create approved tasks
        task = MagicMock()
        task.title = "Test task"
        task.status = "APPROVED"
        task.result = "Research result"
        task.hypotheses = None
        task.evidence_rating = None
        task.contradictions = None
        
        # Reporter agent crashes
        mock_agents['reporter'].return_value.side_effect = Exception("Out of memory")
        
        job_id = str(sample_research_report.id)
        
        with patch('src.worker.tasks.SessionLocal') as mock_session:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.all.return_value = [task]
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_report
            mock_session.return_value = mock_db
            
            aggregate_report(job_id)
            
            # Should be marked as failed
            assert sample_research_report.status == "failed"
            assert "error" in sample_research_report.report


@pytest.mark.integration
class TestTaskStateTransitions:
    """Tests for correct task state machine transitions."""
    
    def test_task_status_progression_pending_to_approved(self, db_session, sample_research_report):
        """Test complete status progression through supervisor."""
        from src.worker.tasks import supervisor_loop
        
        # Test each state transition
        states = [
            ("PENDING", "HYPOTHESIZING_STARTED", "generate_hypotheses_task"),
            ("HYPOTHESIZED", "RESEARCHING_STARTED", "perform_research_task"),
            ("RESEARCHED", "SCORING_STARTED", "score_evidence_task"),
            ("SCORED", "CONTRADICTING_STARTED", "find_contradictions_task"),
            ("CONTRADICTED", "REVIEW_STARTED", "review_task"),
        ]
        
        for initial_status, expected_status, expected_task in states:
            task = MagicMock()
            task.id = uuid4()
            task.job_id = sample_research_report.id
            task.status = initial_status
            
            with patch('src.worker.tasks.SessionLocal') as mock_session, \
                 patch(f'src.worker.tasks.{expected_task}') as mock_task:
                
                mock_db = MagicMock()
                mock_db.query.return_value.filter.return_value.all.return_value = [task]
                mock_session.return_value = mock_db
                
                supervisor_loop(str(sample_research_report.id))
                
                assert task.status == expected_status
                mock_task.delay.assert_called()
    
    def test_all_approved_triggers_report(self, db_session, sample_research_report):
        """Test that when all tasks are approved, report generation starts."""
        from src.worker.tasks import supervisor_loop
        
        tasks = [
            MagicMock(status="APPROVED", id=uuid4()),
            MagicMock(status="APPROVED", id=uuid4()),
            MagicMock(status="APPROVED", id=uuid4())
        ]
        
        with patch('src.worker.tasks.SessionLocal') as mock_session, \
             patch('src.worker.tasks.aggregate_report') as mock_report:
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.all.return_value = tasks
            mock_db.query.return_value.filter.return_value.first.return_value = sample_research_report
            mock_session.return_value = mock_db
            
            supervisor_loop(str(sample_research_report.id))
            
            # Report should be triggered
            mock_report.delay.assert_called_once()
            assert sample_research_report.status == "generating"

