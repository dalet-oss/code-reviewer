import logging
import os
from google import genai
from google.genai import types
from llm_interface import LLMInterface, ModelResult
from config import LOG_CHAR_LIMIT


class GeminiLLM(LLMInterface):
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini")
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    def answer(self, system_prompt: str, user_prompt: str, content: str) -> ModelResult:
        """Generate a response for the given prompts and content."""
        full_input = f"{system_prompt}\n\n{user_prompt}\n\n{content}" if user_prompt else f"{system_prompt}\n\n{content}"
        logging.debug(
            f"Gemini Request:\nModel: {self.model_name}\nContent: {full_input[:LOG_CHAR_LIMIT]}... (truncated)")

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_input,
                config=types.GenerateContentConfig(
                    temperature=0.0  # Maximum consistency
                )
            )
            raw_response = (response.text or "").strip()
            usage = response.usage_metadata
            prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
            completion_tokens = getattr(usage, "candidates_token_count", None) or 0
            total_tokens = getattr(usage, "total_token_count", None) or (prompt_tokens + completion_tokens)
            logging.debug(f"Raw Response:\n{raw_response[:LOG_CHAR_LIMIT]}... (truncated)")            
            return ModelResult(response=raw_response, 
                             total_tokens=total_tokens,
                             prompt_tokens=prompt_tokens,
                             completion_tokens=completion_tokens)
        except Exception as e:
            logging.error(f"Error communicating with Gemini API: {str(e)}")            
            return None
