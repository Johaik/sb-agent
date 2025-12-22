import boto3
import json
from typing import List, Dict, Any, Optional
from .base import LLMProvider
from ..config import Config

class BedrockProvider(LLMProvider):
    def __init__(self, region_name: str = Config.BEDROCK_REGION, profile_name: str = Config.BEDROCK_PROFILE):
        try:
            session = boto3.Session(profile_name=profile_name)
            self.client = session.client("bedrock-runtime", region_name=region_name)
        except Exception:
            self.client = boto3.client("bedrock-runtime", region_name=region_name)
        
        self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        self.embedding_model_id = "amazon.titan-embed-text-v2:0"

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        bedrock_messages = []
        system_prompt = ""

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt += content + "\n"
                continue
            
            if role == "tool":
                # Convert generic tool result to Bedrock format
                bedrock_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id"),
                            "content": content
                        }
                    ]
                })
            elif role == "assistant":
                # Reconstruct assistant message with tool uses if present
                new_content = []
                if content:
                    new_content.append({"type": "text", "text": content})
                
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        new_content.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["input"]
                        })
                
                bedrock_messages.append({
                    "role": "assistant",
                    "content": new_content
                })
            else:
                # User messages
                bedrock_messages.append({
                    "role": role,
                    "content": content
                })
                
        return system_prompt.strip(), bedrock_messages

    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        system_prompt, bedrock_messages = self._convert_messages(messages)
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": bedrock_messages,
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        if tools:
            bedrock_tools = []
            for tool in tools:
                tool_def = tool.copy()
                if "parameters" in tool_def:
                    tool_def["input_schema"] = tool_def.pop("parameters")
                bedrock_tools.append(tool_def)
            body["tools"] = bedrock_tools

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response.get("body").read())
            
            content_blocks = response_body.get("content", [])
            text_content = ""
            tool_calls = []
            
            for block in content_blocks:
                if block["type"] == "text":
                    text_content += block["text"]
                elif block["type"] == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "name": block["name"],
                        "input": block["input"]
                    })
            
            return {
                "content": text_content,
                "tool_calls": tool_calls if tool_calls else None,
                "raw": response_body
            }
            
        except Exception as e:
            print(f"Error invoking Bedrock: {e}")
            raise e

    def get_embedding(self, text: str) -> List[float]:
        body = json.dumps({
            "inputText": text,
            "dimensions": 1024,
            "normalize": True
        })
        
        try:
            response = self.client.invoke_model(
                modelId=self.embedding_model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response.get("body").read())
            return response_body.get("embedding")
        except Exception as e:
            print(f"Error getting embedding from Bedrock: {e}")
            raise e
