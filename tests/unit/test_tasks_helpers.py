"""
Unit tests for helper functions in src/worker/tasks.py
Tests clean_json_string and other utility functions.
"""
import pytest
import json


class TestCleanJsonString:
    """Tests for the clean_json_string helper function."""
    
    @pytest.fixture
    def clean_json_string(self):
        """Import the clean_json_string function."""
        from src.worker.tasks import clean_json_string
        return clean_json_string
    
    def test_clean_simple_json(self, clean_json_string):
        """Test cleaning a simple JSON string."""
        input_str = '{"key": "value"}'
        result = clean_json_string(input_str)
        
        assert result == '{"key": "value"}'
        # Verify it's valid JSON
        parsed = json.loads(result)
        assert parsed["key"] == "value"
    
    def test_clean_json_with_markdown_code_block(self, clean_json_string):
        """Test removing ```json code block markers."""
        input_str = '```json\n{"key": "value"}\n```'
        result = clean_json_string(input_str)
        
        assert result == '{"key": "value"}'
        parsed = json.loads(result)
        assert parsed["key"] == "value"
    
    def test_clean_json_with_plain_code_block(self, clean_json_string):
        """Test removing ``` code block markers without json tag."""
        input_str = '```\n{"key": "value"}\n```'
        result = clean_json_string(input_str)
        
        assert result == '{"key": "value"}'
    
    def test_clean_json_with_only_closing_marker(self, clean_json_string):
        """Test removing only closing ``` marker."""
        input_str = '{"key": "value"}\n```'
        result = clean_json_string(input_str)
        
        assert result == '{"key": "value"}'
    
    def test_clean_json_with_leading_whitespace(self, clean_json_string):
        """Test handling leading whitespace."""
        input_str = '   \n\n{"key": "value"}'
        result = clean_json_string(input_str)
        
        assert result == '{"key": "value"}'
    
    def test_clean_json_with_trailing_whitespace(self, clean_json_string):
        """Test handling trailing whitespace."""
        input_str = '{"key": "value"}   \n\n'
        result = clean_json_string(input_str)
        
        assert result == '{"key": "value"}'
    
    def test_clean_json_array(self, clean_json_string):
        """Test cleaning a JSON array."""
        input_str = '```json\n["task1", "task2", "task3"]\n```'
        result = clean_json_string(input_str)
        
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 3
    
    def test_clean_json_complex_object(self, clean_json_string):
        """Test cleaning a complex nested JSON object."""
        complex_json = {
            "summary": "Test summary",
            "key_findings": ["Finding 1", "Finding 2"],
            "details": {
                "section1": "Content",
                "section2": {
                    "nested": True
                }
            }
        }
        input_str = f'```json\n{json.dumps(complex_json)}\n```'
        result = clean_json_string(input_str)
        
        parsed = json.loads(result)
        assert parsed["summary"] == "Test summary"
        assert len(parsed["key_findings"]) == 2
    
    def test_clean_json_with_multiline_content(self, clean_json_string):
        """Test cleaning JSON with multiline string values."""
        input_str = '''```json
{
    "text": "Line 1\\nLine 2\\nLine 3",
    "approved": true
}
```'''
        result = clean_json_string(input_str)
        
        parsed = json.loads(result)
        assert parsed["approved"] is True
    
    def test_clean_json_empty_string(self, clean_json_string):
        """Test handling empty string."""
        result = clean_json_string("")
        assert result == ""
    
    def test_clean_json_only_whitespace(self, clean_json_string):
        """Test handling whitespace-only string."""
        result = clean_json_string("   \n\n   ")
        assert result == ""
    
    def test_clean_json_no_markers(self, clean_json_string):
        """Test JSON without any markers."""
        input_str = '{"approved": false, "feedback": "Needs more detail"}'
        result = clean_json_string(input_str)
        
        assert result == input_str
        parsed = json.loads(result)
        assert parsed["approved"] is False


class TestHelperFunctions:
    """Tests for other helper functions in tasks module."""
    
    def test_calculate_progress_completed(self):
        """Test progress calculation for completed job."""
        from unittest.mock import MagicMock
        from src.api.main import calculate_progress
        
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = []
        
        progress, phase = calculate_progress("job-123", mock_db, "completed")
        
        assert progress == 100
        assert phase == "reporting"
    
    def test_calculate_progress_failed(self):
        """Test progress calculation for failed job."""
        from unittest.mock import MagicMock
        from src.api.main import calculate_progress
        
        mock_db = MagicMock()
        
        progress, phase = calculate_progress("job-123", mock_db, "failed")
        
        assert progress == 0
        assert phase == "failed"
    
    def test_calculate_progress_pending(self):
        """Test progress calculation for pending job."""
        from unittest.mock import MagicMock
        from src.api.main import calculate_progress
        
        mock_db = MagicMock()
        
        progress, phase = calculate_progress("job-123", mock_db, "pending")
        
        assert progress == 0
        assert phase == "enriching"
    
    def test_calculate_progress_no_tasks(self):
        """Test progress calculation when no tasks exist yet."""
        from unittest.mock import MagicMock
        from src.api.main import calculate_progress
        
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = []
        
        progress, phase = calculate_progress("job-123", mock_db, "processing")
        
        assert progress == 10
        assert phase == "planning"
    
    def test_calculate_progress_partial_tasks(self):
        """Test progress calculation with some tasks completed."""
        from unittest.mock import MagicMock
        from src.api.main import calculate_progress
        
        # Create mock tasks (5 total, 2 approved)
        mock_tasks = []
        for i in range(5):
            task = MagicMock()
            task.status = "APPROVED" if i < 2 else "PENDING"
            mock_tasks.append(task)
        
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = mock_tasks
        
        progress, phase = calculate_progress("job-123", mock_db, "processing")
        
        # 2/5 tasks = 40% of 70% = 28% + 20% base = 48%
        assert progress == 48
        assert phase == "researching"
    
    def test_calculate_progress_all_tasks_done(self):
        """Test progress calculation when all tasks are approved."""
        from unittest.mock import MagicMock
        from src.api.main import calculate_progress
        
        # All 3 tasks approved
        mock_tasks = []
        for i in range(3):
            task = MagicMock()
            task.status = "APPROVED"
            mock_tasks.append(task)
        
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = mock_tasks
        
        progress, phase = calculate_progress("job-123", mock_db, "processing")
        
        assert progress == 90
        assert phase == "reporting"
    
    def test_calculate_progress_capped_at_99(self):
        """Test progress never exceeds 99 during processing."""
        from unittest.mock import MagicMock
        from src.api.main import calculate_progress
        
        # Simulate edge case
        mock_tasks = []
        for i in range(100):  # Many tasks
            task = MagicMock()
            task.status = "APPROVED" if i < 95 else "PENDING"
            mock_tasks.append(task)
        
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = mock_tasks
        
        progress, phase = calculate_progress("job-123", mock_db, "processing")
        
        # Should still show 90% when all approved (which it isn't here)
        # or stay below 99
        assert progress <= 99


class TestJsonParsingEdgeCases:
    """Test JSON parsing edge cases in tasks."""
    
    @pytest.fixture
    def clean_json_string(self):
        """Import the clean_json_string function."""
        from src.worker.tasks import clean_json_string
        return clean_json_string
    
    def test_parse_tasks_from_llm_response(self, clean_json_string):
        """Test parsing tasks array from LLM response."""
        llm_response = '''```json
["Research the history of quantum computing",
 "Analyze current quantum computing applications",
 "Investigate future predictions for quantum computing"]
```'''
        
        cleaned = clean_json_string(llm_response)
        tasks = json.loads(cleaned)
        
        assert len(tasks) == 3
        assert "quantum computing" in tasks[0].lower()
    
    def test_parse_hypothesis_from_llm_response(self, clean_json_string):
        """Test parsing hypothesis JSON from LLM response."""
        llm_response = '''```json
{
    "hypotheses": [
        {"statement": "Quantum computing will break RSA by 2030", "confidence": "medium", "reasoning": "Based on current progress"}
    ]
}
```'''
        
        cleaned = clean_json_string(llm_response)
        data = json.loads(cleaned)
        
        assert "hypotheses" in data
        assert len(data["hypotheses"]) == 1
        assert data["hypotheses"][0]["confidence"] == "medium"
    
    def test_parse_critic_response(self, clean_json_string):
        """Test parsing critic JSON response."""
        llm_response = '''```json
{
    "approved": true,
    "feedback": "Research is comprehensive and well-documented."
}
```'''
        
        cleaned = clean_json_string(llm_response)
        review = json.loads(cleaned)
        
        assert review["approved"] is True
        assert "comprehensive" in review["feedback"]
    
    def test_parse_evidence_rating(self, clean_json_string):
        """Test parsing evidence rating JSON."""
        llm_response = '''```json
{
    "relevance_score": 8,
    "credibility_score": 7,
    "analysis": "Good coverage of the topic.",
    "weak_points": ["Limited peer-reviewed sources"]
}
```'''
        
        cleaned = clean_json_string(llm_response)
        rating = json.loads(cleaned)
        
        assert rating["relevance_score"] == 8
        assert rating["credibility_score"] == 7
        assert len(rating["weak_points"]) == 1
    
    def test_parse_contradiction_response(self, clean_json_string):
        """Test parsing contradiction seeker response."""
        llm_response = '''```json
{
    "contradictions_found": false,
    "details": []
}
```'''
        
        cleaned = clean_json_string(llm_response)
        data = json.loads(cleaned)
        
        assert data["contradictions_found"] is False
        assert data["details"] == []

