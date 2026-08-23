"""
Week 1: Minimal Ollama client — health check only.

Full LLM integration (generate, stream, RAG prompts) will be added in Week 5.
"""

import logging
from typing import Any, Dict

import httpx

from src.config import Settings
from src.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama local LLM service.

    Week 1: Health check only.
    Week 5: Full LLM integration with generate, stream, and RAG prompts.
    """

    def __init__(self, settings: Settings):
        """Initialize Ollama client with settings."""
        self.base_url = settings.ollama_host
        self.timeout = httpx.Timeout(float(settings.ollama_timeout))

    async def health_check(self) -> Dict[str, Any]:
        """Check if Ollama service is healthy and responding.

        Returns:
            Dictionary with health status information
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Check version endpoint for health
                response = await client.get(f"{self.base_url}/api/version")

                if response.status_code == 200:
                    version_data = response.json()
                    return {
                        "status": "healthy",
                        "message": "Ollama service is running",
                        "version": version_data.get("version", "unknown"),
                    }
                else:
                    raise OllamaException(f"Ollama returned status {response.status_code}")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Ollama health check failed: {str(e)}")
