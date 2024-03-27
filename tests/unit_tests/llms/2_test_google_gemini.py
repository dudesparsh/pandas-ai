"""Unit tests for the openai LLM class"""
import re

import pytest
from google import generativeai

from pandasai.exceptions import APIKeyNotFoundError
from pandasai.llm import GoogleGemini
from pandasai.prompts import AbstractPrompt

class MockedCompletion:
    def __init__(self, result: str):
        self.result = result

class TestGoogleGemini:
    """Unit tests for the Google Gemini LLM class"""

    @pytest.fixture
    def prompt(self):
        class MockAbstractPrompt(AbstractPrompt):
            template: str = "Hello"

        return MockAbstractPrompt()

    def test_type_without_token(self):
        with pytest.raises(APIKeyNotFoundError):
            GoogleGemini(api_key="")

    def test_type_with_token(self):
        assert GoogleGemini(api_key="test").type == "google-gemini"

    def test_params_setting(self):
        llm = GoogleGemini(
            api_key="test",
            model="models/gemini-pro",
            generation_config ={
            "temperature": 0.5,
            "top_p":1.0,
            "top_k":50,
            "max_output_tokens":64
            }
        )

        assert llm.model == "models/gemini-pro"
        assert llm.generation_config['temperature'] == 0.5
        assert llm.generation_config['top_p'] == 1.0
        assert llm.generation_config['top_k'] == 50
        assert llm.generation_config['max_output_tokens'] == 64

    def test_validations(self, prompt):
        with pytest.raises(
            ValueError, match=re.escape("temperature must be in the range [0.0, 1.0]")
        ):
            GoogleGemini(api_key="test", temperature=-1).call(prompt, "World")

        with pytest.raises(
            ValueError, match=re.escape("temperature must be in the range [0.0, 1.0]")
        ):
            GoogleGemini(api_key="test", temperature=1.1).call(prompt, "World")

        with pytest.raises(
            ValueError, match=re.escape("top_p must be in the range [0.0, 1.0]")
        ):
            GoogleGemini(api_key="test", top_p=-1).call(prompt, "World")

        with pytest.raises(
            ValueError, match=re.escape("top_p must be in the range [0.0, 1.0]")
        ):
            GoogleGemini(api_key="test", top_p=1.1).call(prompt, "World")

        with pytest.raises(
            ValueError, match=re.escape("top_k must be in the range [0.0, 100.0]")
        ):
            GoogleGemini(api_key="test", top_k=-100).call(prompt, "World")

        with pytest.raises(
            ValueError, match=re.escape("top_k must be in the range [0.0, 100.0]")
        ):
            GoogleGemini(api_key="test", top_k=110).call(prompt, "World")

        with pytest.raises(
            ValueError, match=re.escape("max_output_tokens must be greater than zero")
        ):
            GoogleGemini(api_key="test", top_k=0).call(prompt, "World")

        with pytest.raises(ValueError, match=re.escape("model is required.")):
            GoogleGemini(api_key="test", model="").call(prompt, "World")

    def test_text_generation(self, mocker):
        llm = GoogleGemini(api_key="test")
        expected_text = "This is the expected text."
        expected_response = MockedCompletion(expected_text)
        mocker.patch.object(
            generativeai, "generate_text", return_value=expected_response
        )

        result = llm._generate_text("Hi")
        assert result == expected_text

    def test_call(self, mocker, prompt):
        llm = GoogleGemini(api_key="test")
        expected_text = "This is the expected text."
        expected_response = MockedCompletion(expected_text)
        mocker.patch.object(
            generativeai, "generate_text", return_value=expected_response
        )

        result = llm.call(instruction=prompt, suffix="!")
        assert result == expected_text
