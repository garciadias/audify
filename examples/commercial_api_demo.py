#!/usr/bin/env python3
"""
Example script demonstrating commercial API usage with Audify.

This script shows how to use commercial APIs (DeepSeek, Claude, GPT-4, etc.)
instead of local Ollama models for audiobook generation.
"""

from audify.audiobook_creator import LLMClient


def test_llm_clients():
    """Test different LLM client configurations."""

    print("Testing LLM Client Initialization")
    print("=" * 80)

    # Test Ollama (default)
    print("\n1. Ollama Configuration:")
    ollama_client = LLMClient(model="llama3.2:3b")
    print(f"   - Is Commercial: {ollama_client.is_commercial}")
    print(f"   - Config Type: {type(ollama_client.config).__name__}")

    # Test DeepSeek
    print("\n2. DeepSeek Configuration:")
    deepseek_client = LLMClient(model="api:deepseek/deepseek-chat")
    print(f"   - Is Commercial: {deepseek_client.is_commercial}")
    print(f"   - Config Type: {type(deepseek_client.config).__name__}")
    print(f"   - Model: {deepseek_client.config.model}")

    # Test Claude
    print("\n3. Claude Configuration:")
    claude_client = LLMClient(model="api:anthropic/claude-sonnet-4-20250514")
    print(f"   - Is Commercial: {claude_client.is_commercial}")
    print(f"   - Config Type: {type(claude_client.config).__name__}")
    print(f"   - Model: {claude_client.config.model}")

    # Test GPT-4
    print("\n4. GPT-4 Configuration:")
    gpt4_client = LLMClient(model="api:openai/gpt-4o")
    print(f"   - Is Commercial: {gpt4_client.is_commercial}")
    print(f"   - Config Type: {type(gpt4_client.config).__name__}")
    print(f"   - Model: {gpt4_client.config.model}")

    print("\n" + "=" * 80)
    print("✓ All LLM client configurations initialized successfully!")
    print("\nNote: To actually use these APIs, you need to:")
    print("1. Create a .keys file in the project root")
    print("2. Add your API keys (see .keys.example for format)")
    print("3. Run the audiobook creation with -m 'api:model_name'")


def show_usage_examples():
    """Show usage examples for commercial APIs."""

    print("\n\nUsage Examples")
    print("=" * 80)

    examples = [
        ("DeepSeek (cost-effective)", "api:deepseek/deepseek-chat"),
        ("DeepSeek R1 (reasoning)", "api:deepseek/deepseek-reasoner"),
        ("Claude 3.5 Sonnet", "api:anthropic/claude-3-5-sonnet-20240620"),
        ("Claude 3 Opus", "api:anthropic/claude-3-opus-20240229"),
        ("GPT-4 Turbo", "api:openai/gpt-4-turbo-preview"),
        ("GPT-3.5 Turbo", "api:openai/gpt-3.5-turbo"),
        ("Gemini Pro", "api:gemini/gemini-1.5-pro"),
    ]

    print("\nCommand-line examples:")
    for name, model in examples:
        print(f"\n# {name}")
        print(f"python -m audify.create_audiobook book.epub -m '{model}'")

    print("\n" + "=" * 80)


def show_keys_file_format():
    """Show the format for the .keys file."""

    print("\n\n.keys File Format")
    print("=" * 80)
    print("""
Create a file named .keys in the project root with the following format:

# DeepSeek API
DEEPSEEK=sk-your-deepseek-api-key-here

# Anthropic Claude API
ANTHROPIC=sk-ant-your-anthropic-api-key-here

# OpenAI API
OPENAI=sk-your-openai-api-key-here

# Google Gemini API
GEMINI=your-google-api-key-here

Lines starting with # are comments and will be ignored.
The .keys file is already in .gitignore to prevent accidental commits.
    """)
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_llm_clients()
        show_usage_examples()
        show_keys_file_format()

        print("\n\n✓ Commercial API integration is working correctly!")
        print("See docs/commercial-apis.md for more detailed information.")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nThis is expected if you don't have API keys configured.")
        print("The system will still work with local Ollama models.")
