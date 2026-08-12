import json
import logging
import httpx
from typing import List, Dict, Any, Optional
from runtime.gpu_broker import gpu_broker

logger = logging.getLogger("lm_studio_client")

class LMStudioClient:
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", model_name: str = "google/gemma-4-12b"):
        self.base_url = base_url
        self.model_name = model_name

    async def adjudicate_cut_decisions(
        self,
        words: List[Dict[str, Any]],
        aggressiveness: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Send transcript words to LM Studio for intelligent cut adjudication.
        Returns list of decisions: [{"word_id": "w_0", "keep": bool, "reason": str}]
        """
        if not words:
            return []

        await gpu_broker.acquire_lease("lm_studio", required_vram_mb=7000.0)
        try:
            # Prepare compact representation for LLM prompt
            prompt_items = [
                {"id": w.get("id"), "text": w.get("text"), "disfluency": w.get("disfluency", False)}
                for w in words
            ]

            system_prompt = (
                "You are an expert video editor AI. Analyze transcript words and decide which filler, "
                "repeated, or redundant words should be cut (keep=false) or preserved (keep=true). "
                "Return a valid JSON array of objects with fields: 'word_id', 'keep', 'reason'."
            )

            user_prompt = f"Transcript items:\n{json.dumps(prompt_items[:100])}\nAggressiveness: {aggressiveness}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.2,
                        "response_format": {"type": "json_object"}
                    }
                )

                if resp.status_code == 200:
                    result = resp.json()
                    content = result["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, dict) and "decisions" in parsed:
                        return parsed["decisions"]

            logger.warning("LM Studio call returned non-200 or unexpected structure. Fallback to default disfluency rule.")
            return [{"word_id": w["id"], "keep": not w.get("disfluency", False)} for w in words]
        except Exception as e:
            logger.warning(f"LM Studio connection/inference error: {e}. Fallback active.")
            return [{"word_id": w["id"], "keep": not w.get("disfluency", False)} for w in words]
        finally:
            await gpu_broker.release_lease("lm_studio")

lm_studio_client = LMStudioClient()
