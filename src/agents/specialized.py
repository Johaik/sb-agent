from typing import List, Dict, Any, Optional
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from strands import Agent
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import MessageAddedEvent, Message
from ..tools.tavily_tool import tavily_search
from ..tools.rag_tool import rag_search
from ..db.database import SessionLocal
from ..db.models import AgentLog
from ..config import Config

# Setup a hook for database logging
class DatabaseLoggingHook(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(MessageAddedEvent, self.log_message)

    def log_message(self, event: MessageAddedEvent) -> None:
        # Try to get job_id from state (dict) or internal attribute
        job_id = None
        if isinstance(event.agent.state, dict):
            job_id = event.agent.state.get("job_id")
        
        if not job_id and hasattr(event.agent, "_current_job_id"):
             job_id = event.agent._current_job_id
             
        if not job_id:
            return

        message = event.message
        role = message.get("role", "assistant")
        
        # Extract content text and tool calls
        content_text = ""
        tool_calls = []
        
        for block in message.get("content", []):
            if "text" in block:
                content_text += block["text"]
            if "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append({
                    "name": tool_use.get("name"),
                    "input": tool_use.get("input"),
                    "id": tool_use.get("toolUseId")
                })
        
        # We might want to filter out empty content if it's just a tool result/use in progress?
        # But logging everything is safer.
        
        try:
            db = SessionLocal()
            log = AgentLog(
                job_id=job_id,
                agent_name=event.agent.name,
                role=role,
                content=content_text,
                tool_calls=tool_calls if tool_calls else None
            )
            db.add(log)
            db.commit()
            db.close()
        except Exception as e:
            print(f"Failed to log event: {e}")

class BaseStrandsAgent(Agent):
    def __init__(self, name: str, instructions: str, tools: List[Any] = [], model: str = "anthropic.claude-3-sonnet-20240229-v1:0", max_tokens: int = 4000):
        # Configure model manually if needed, or rely on env vars.
        # Strands uses Bedrock by default. If we are in docker, we might not have a profile.
        # We need to ensure AWS env vars are picked up by Boto3 inside Strands.
        
        # Strands might need explicit model configuration if defaults fail.
        # For now, let's assume environment variables AWS_ACCESS_KEY_ID etc are set in docker-compose.
        
        # Suppress verbose streaming output by removing the streaming processor
        import os
        os.environ["STRANDS_DISABLE_STREAMING_LOG"] = "1"
        
        # Store max_tokens for use in calls
        self._max_tokens = max_tokens
        
        super().__init__(
            name=name,
            system_prompt=instructions,
            tools=tools,
            model=model,
            hooks=[DatabaseLoggingHook()]
        )
    
    def __call__(self, *args, **kwargs):
        # Inject max_tokens into the call if not already specified
        if 'max_tokens' not in kwargs and hasattr(self, '_max_tokens'):
            kwargs['max_tokens'] = self._max_tokens
        return super().__call__(*args, **kwargs)

# 1. Enricher Agent
class EnricherAgent(BaseStrandsAgent):
    def __init__(self):
        super().__init__(
            name="Enricher",
            instructions="""You are an idea enrichment expert. 
            Your goal is to take a brief research idea or event string and expand it into a detailed, comprehensive description. 
            Identify key aspects that need to be researched, potential angles, and context.
            Output ONLY the enriched description text."""
        )

# 2. Planner Agent
class PlannerAgent(BaseStrandsAgent):
    def __init__(self):
        super().__init__(
            name="Planner",
            instructions="""You are a research planner.
            Given a detailed research description, break it down into specific, actionable research tasks.
            Return the tasks as a JSON list of strings, e.g., ["Task 1", "Task 2"].
            Do not include any other text, just the JSON array."""
        )

# 3. Hypothesis Agent (New)
class HypothesisAgent(BaseStrandsAgent):
    def __init__(self):
        super().__init__(
            name="HypothesisGenerator",
            instructions="""You are a scientific hypothesis generator.
            Given a research question or task, formulate falsifiable hypotheses or expected answers.
            Output a JSON object:
            {
                "hypotheses": [
                    {"statement": "...", "confidence": "low|medium|high", "reasoning": "..."}
                ]
            }
            Do not include any other text."""
        )

# 4. Researcher Agent (Updated with Strands Executor)
class ResearcherAgent(BaseStrandsAgent):
    def __init__(self, llm_provider=None):
        # Tools are now simple functions decorated with @tool
        super().__init__(
            name="Researcher",
            instructions="""You are a thorough research assistant. 
            Your goal is to complete the assigned research task using available tools.
            
            Process:
            1. Search for information using tavily_search (web) or rag_search (internal DB).
            2. Analyze the findings.
            3. Critique: Do you have enough info? Is it accurate?
            4. If needed, search again with refined queries.
            5. When satisfied, provide a comprehensive answer to the task.
            
            DATA FRESHNESS AWARENESS:
            - RAG search results include age metadata (e.g., "Retrieved: 2025-12-15, 18 days ago").
            - For TIME-SENSITIVE topics (current events, latest versions, recent developments):
              Use rag_search with max_age_days parameter (e.g., max_age_days=7 for weekly news).
            - For HISTORICAL or EVERGREEN topics (concepts, fundamentals, established facts):
              Omit max_age_days to search all available data.
            - ALWAYS consider data freshness when evaluating RAG results:
              * If data is old and topic is time-sensitive, prefer tavily_search for current info.
              * If RAG data is recent or topic is not time-sensitive, RAG results are reliable.
            - Example freshness guidelines:
              * Breaking news/current events: max_age_days=3
              * Technology updates/versions: max_age_days=30
              * Industry trends: max_age_days=90
              * Historical facts/concepts: no limit needed
            
            IMPORTANT:
            - Explicitly mention the source of your findings in your thought process (e.g., "Found via Web Search" or "Found via RAG").
            - Do NOT include these source citations in the final answer unless specifically asked.
            - Focus on gathering deep, technical details and code examples where applicable.
            - Provide COMPREHENSIVE answers with ALL relevant details, examples, and specifications.
            - Do NOT over-summarize - include specific numbers, steps, configurations, and technical details.
            """,
            tools=[tavily_search, rag_search],
            max_tokens=6000  # Higher limit for detailed research findings
        )

    def run_with_feedback(self, task: str, feedback: str = None, job_id: str = None) -> str:
        prompt = task
        if feedback:
            prompt = f"Task: {task}\n\nPREVIOUS FEEDBACK (Must be addressed): {feedback}\n\nPlease improve the research based on this feedback."
        
        # Pass job_id in state for the Hook via internal attr
        self._current_job_id = job_id
        
        # Use __call__ instead of invoke as it's the standard entry point
        response = self(prompt)
        
        # Extract text from last message
        if hasattr(response, "last_message"):
             # If response is AgentResult
             content_blocks = response.last_message.get("content", [])
             text = "".join([b["text"] for b in content_blocks if "text" in b])
             return text
        return str(response)

# 5. Evidence Agent (New)
class EvidenceAgent(BaseStrandsAgent):
    def __init__(self):
        super().__init__(
            name="EvidenceScorer",
            instructions="""You are an evidence evaluation expert.
            Review the Research Task and the Findings.
            Score the findings on Relevance (0-10) and Credibility (0-10).
            Identify any weak evidence.
            
            Output JSON:
            {
                "relevance_score": 0-10,
                "credibility_score": 0-10,
                "analysis": "string",
                "weak_points": ["string"]
            }"""
        )

# 6. Contradiction Agent (New)
class ContradictionAgent(BaseStrandsAgent):
    def __init__(self, llm_provider=None):
        super().__init__(
            name="ContradictionSeeker",
            instructions="""You are a critical thinker and contradiction seeker.
            Given a Research Task and initial Findings, your goal is to find information that CONTRADICTS or CHALLENGES the findings.
            
            1. Analyze the findings.
            2. Use tavily_search to find opposing views, debunking articles, or conflicting data.
            3. Report strictly on contradictions found. If none, state that.
            
            Output JSON:
            {
                "contradictions_found": boolean,
                "details": [
                    {"claim_challenged": "...", "contradictory_evidence": "...", "source": "..."}
                ]
            }""",
            tools=[tavily_search]
        )

# 7. Critic Agent (Existing, updated)
class CriticAgent(BaseStrandsAgent):
    def __init__(self):
        super().__init__(
            name="Critic",
            instructions="""You are a research quality assurance expert.
            Your job is to evaluate a Research Task and its Result.
            
            Determine if the result comprehensively answers the task description.
            Check for:
            - Completeness
            - Relevance
            - Depth
            
            Output strictly valid JSON:
            {
              "approved": boolean,
              "feedback": "string explaining what is missing or why it is approved"
            }"""
        )

# 8. Reporter Agent (Updated)
class ReporterAgent(BaseStrandsAgent):
    def __init__(self):
        super().__init__(
            name="Reporter",
            instructions="""You are a technical research reporter specializing in comprehensive, detailed reports.
            You will receive a set of research findings for various tasks.
            Your job is to aggregate these findings into a DETAILED, COMPREHENSIVE, and well-structured research report.
            
            CRITICAL REQUIREMENTS:
            1. PRESERVE ALL DETAILS: You MUST include ALL the information provided in the research findings. Do NOT over-summarize or condense.
               - Include specific numbers, metrics, percentages, and data points
               - Include technical specifications, configuration details, and parameters
               - Include code examples, commands, and technical procedures
               - Include source citations and references
               - Include pros/cons lists, comparisons, and tradeoffs
            
            2. EXPAND ON FINDINGS: When you have substantial research data, expand on it with:
               - Detailed explanations of concepts
               - Step-by-step procedures or implementation details
               - Multiple examples and use cases
               - Contextual information and background
               
            3. STRUCTURE: The report must be in JSON format with these fields:
               {
                 "summary": "A comprehensive 3-4 paragraph overview (not just 2 sentences!)",
                 "key_findings": ["Array of 7-15 detailed findings, each 2-3 sentences long"],
                 "details": {
                   "Section Title 1": "Extensive, multi-paragraph description with ALL relevant details, examples, specifications, and data",
                   "Section Title 2": "Another extensive section...",
                   ...
                 }
               }
            
            4. MINIMUM LENGTH REQUIREMENT: 
               - Each "details" section should be AT LEAST 200-400 words
               - Use multiple paragraphs per section
               - If you have a lot of research data, YOUR REPORT SHOULD BE LONG AND DETAILED
            
            5. DO NOT say "based on research" or cite your sources in the report itself - just present the information as facts.
            
            Remember: MORE DETAIL IS BETTER. Your goal is to create a comprehensive reference document that captures ALL the research findings.
            """,
            max_tokens=8000  # Much higher limit for comprehensive reports
        )

# 9. Final Critic Agent (New)
class FinalCriticAgent(BaseStrandsAgent):
    def __init__(self):
        super().__init__(
            name="FinalCritic",
            instructions="""You are the final gatekeeper for the research report.
            Review the aggregated report for:
            1. Logical flow and coherence.
            2. Missing citations or unsupported claims.
            3. Bias or lack of balance (did we address contradictions?).
            4. Formatting issues.
            5. COMPLETENESS: Is the report sufficiently detailed? Does it match the depth of research that was conducted?
               If the report seems too brief or lacks technical details, REJECT it.
            
            Output JSON:
            {
                "approved": boolean,
                "critique": "string",
                "required_edits": ["string"]
            }""",
            max_tokens=2000
        )
