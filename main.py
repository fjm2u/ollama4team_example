from typing import Any, Dict, Iterator, List, Optional

import requests
from langchain_community.llms import Ollama


class Ollama4Team(Ollama):
    """Ollama4Team is designed for team usage of Ollama.
    Ref: https://github.com/langchain-ai/langchain/blob/cccc8fbe2fe59bde0846875f67aa046aeb1105a3/libs/community/langchain_community/llms/ollama.py

    Example:

        .. code-block:: python

            model = Ollama4Team(api_key="", model="llama2:13b")
            result = model.invoke([HumanMessage(content="hello")])
    """

    """The default parameters for the Ollama API."""
    password: str
    base_url: str

    def _create_stream(
        self,
        api_url: str,
        payload: Any,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop

        params = self._default_params

        for key in self._default_params:
            if key in kwargs:
                params[key] = kwargs[key]

        if "options" in kwargs:
            params["options"] = kwargs["options"]
        else:
            params["options"] = {
                **params["options"],
                "stop": stop,
                **{k: v for k, v in kwargs.items() if k not in self._default_params},
            }

        if payload.get("messages"):
            request_payload = {"messages": payload.get("messages", []), **params}
        else:
            request_payload = {
                "prompt": payload.get("prompt"),
                "images": payload.get("images", []),
                **params,
            }

        response = requests.post(
            url=api_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Basic {self.password}",
                **(self.headers if isinstance(self.headers, dict) else {}),
            },
            json=request_payload,
            stream=True,
            timeout=self.timeout,
        )
        response.encoding = "utf-8"
        if response.status_code != 200:
            if response.status_code == 404:
                raise self.OllamaEndpointNotFoundError(
                    "Ollama call failed with status code 404. "
                    "Maybe your model is not found "
                    f"and you should pull the model with `ollama pull {self.model}`."
                )
            else:
                optional_detail = response.text
                raise ValueError(
                    f"Ollama call failed with status code {response.status_code}."
                    f" Details: {optional_detail}"
                )
        return response.iter_lines(decode_unicode=True)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "Ollama4Team"


llm = Ollama4Team(model="llama2:13b", base_url="https://example.com", password="YOUR_API_KEY")
llm.invoke("The first man on the moon was ... think step by step")
