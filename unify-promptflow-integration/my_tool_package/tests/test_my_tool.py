import unittest
from unittest.mock import patch, MagicMock
from promptflow.connections import CustomConnection
from my_tool_package.tools.my_tool import optimize_llm, benchmark_models


class TestTool(unittest.TestCase):

    @patch('my_tool_package.tools.my_tool.requests.post')
    def test_optimize_llm(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "optimal_model": "gpt-3.5-turbo",
            "choices": [
                {"message": {"content": "Translation Result"}}
            ]
        }
        mock_post.return_value = mock_response

        my_custom_connection = CustomConnection(
            {
                "api_key": "my-api-key",
                "api_secret": "my-api-secret",
                "api_url": "my-api-url"
            }
        )
        result = optimize_llm(
            my_custom_connection,
            prompt="Translate 'Hello' to French",
            constraints={"max_latency": 1000, "min_quality": 0.7}
        )
        self.assertEqual(result, "Translation Result")

    @patch('my_tool_package.tools.my_tool.requests.post')
    def test_benchmark_models(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": {"gpt-3.5-turbo": 0.8, "gpt-4": 0.9, "claude-2": 0.85}
        }
        mock_post.return_value = mock_response

        my_custom_connection = CustomConnection(
            {
                "api_key": "my-api-key",
                "api_secret": "my-api-secret",
                "api_url": "my-api-url"
            }
        )
        result = benchmark_models(
            my_custom_connection,
            prompt_set=[
                "Translate 'Hello' to Spanish",
                "Translate 'Goodbye' to German"
            ],
            models=["gpt-3.5-turbo", "gpt-4", "claude-2"]
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(
            result,
            {"results": {"gpt-3.5-turbo": 0.8, "gpt-4": 0.9, "claude-2": 0.85}}
        )


if __name__ == "__main__":
    unittest.main()
