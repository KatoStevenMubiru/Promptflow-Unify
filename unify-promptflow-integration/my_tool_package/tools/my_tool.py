from promptflow import tool
from promptflow.connections import CustomConnection
import requests


class UnifyTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.unify.ai/v0"

    def select_optimal_model(self, constraints: dict) -> str:
        response = requests.post(
            f"{self.base_url}/model/optimize",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=constraints
        )
        return response.json().get("optimal_model")

    def generate(self, prompt: str, model: str) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        choices = response.json().get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    def benchmark(self, prompt_set: list, models: list) -> dict:
        response = requests.post(
            f"{self.base_url}/benchmark",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"prompts": prompt_set, "models": models}
        )
        return response.json()


@tool
def optimize_llm(
    connection: CustomConnection,
    prompt: str,
    constraints: dict
) -> str:
    unify_tool = UnifyTool(api_key=connection["api_key"])
    optimal_model = unify_tool.select_optimal_model(constraints)
    response = unify_tool.generate(prompt, model=optimal_model)
    return response


@tool
def benchmark_models(
    connection: CustomConnection,
    prompt_set: list,
    models: list
) -> dict:
    unify_tool = UnifyTool(api_key=connection["api_key"])
    results = unify_tool.benchmark(prompt_set, models)
    return results
