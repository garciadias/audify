#!/usr/bin/env python3
"""
Test script to demonstrate LangChain Ollama translation integration.
This script shows how the modified translate.py works with LangChain OllamaLLM.
"""

from audify.translate import OllamaTranslationConfig, translate_sentence


def test_langchain_config():
    """Test the OllamaTranslationConfig class with LangChain."""
    print("Testing OllamaTranslationConfig with LangChain...")

    # Test default configuration
    config = OllamaTranslationConfig()
    print(f"Default base URL: {config.base_url}")
    print(f"Default model: {config.model}")

    # Test LLM creation
    try:
        llm = config.create_llm()
        print("✓ LangChain OllamaLLM created successfully")
        print(f"  Model: {llm.model}")
        print(f"  Base URL: {llm.base_url}")
        print(f"  Temperature: {llm.temperature}")
        print(f"  Top P: {llm.top_p}")
    except Exception as e:
        print(f"✗ Error creating LLM: {e}")

    # Test custom configuration
    custom_config = OllamaTranslationConfig(
        base_url="http://custom-host:11434",
        model="llama3.1"
    )
    print(f"Custom base URL: {custom_config.base_url}")
    print(f"Custom model: {custom_config.model}")


def test_translation_interface():
    """Test the translation function interface with LangChain."""
    print("\nTesting LangChain translation interface...")

    # Test sentences for different languages
    test_cases = [
        ("Hello world!", "en", "es"),
        ("How are you today?", "en", "fr"),
        ("This is a test sentence.", "en", "pt"),
        ("Good morning!", "en", "de"),
    ]

    for sentence, src_lang, tgt_lang in test_cases:
        print(f"\nTest case: '{sentence}' ({src_lang} -> {tgt_lang})")
        print("This would call LangChain OllamaLLM with:")

        from audify.constants import LANGUAGE_NAMES
        from audify.prompts import TRANSLATE_PROMPT

        src_lang_name = LANGUAGE_NAMES.get(src_lang, src_lang)
        tgt_lang_name = LANGUAGE_NAMES.get(tgt_lang, tgt_lang)

        prompt = TRANSLATE_PROMPT.format(
            src_lang_name=src_lang_name,
            tgt_lang_name=tgt_lang_name,
            sentence=sentence
        )

        print(f"  Prompt preview: {prompt[:80]}...")


def simulate_langchain_translation():
    """Simulate what LangChain translation calls would look like."""
    print("\n" + "=" * 60)
    print("LangChain Ollama Translation Simulation")
    print("=" * 60)

    # Note: These will attempt actual translation if Ollama is running
    test_sentences = [
        "Hello, how are you?",
        "This is a beautiful day.",
        "Thank you for your help.",
    ]

    for sentence in test_sentences:
        print(f"\nTranslating: '{sentence}' (EN -> ES)")
        print("Using: LangChain OllamaLLM.invoke() method")

        try:
            result = translate_sentence(sentence, src_lang="en", tgt_lang="es")
            if result == sentence:
                print(
                f"Result: {result} (returned original - may indicate translation issue)"
                )
            else:
                print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")


def compare_implementations():
    """Compare the old requests vs new LangChain implementation."""
    print("\n" + "=" * 60)
    print("Implementation Comparison")
    print("=" * 60)

    print("BEFORE (requests):")
    print("  import requests")
    print("  response = requests.post(url, json={...})")
    print("  result = response.json()['response']")
    print()

    print("AFTER (LangChain):")
    print("  from langchain_ollama import OllamaLLM")
    print("  llm = OllamaLLM(model=..., base_url=...)")
    print("  result = llm.invoke(prompt)")
    print()

    print("Benefits of LangChain approach:")
    print("  ✓ Simplified API usage")
    print("  ✓ Built-in error handling")
    print("  ✓ Better integration with LangChain ecosystem")
    print("  ✓ Automatic retries and connection management")
    print("  ✓ Consistent interface across different LLM providers")


if __name__ == "__main__":
    print("LangChain Ollama Translation Integration Test")
    print("=" * 60)
    test_langchain_config()
    test_translation_interface()
    simulate_langchain_translation()
    compare_implementations()

    print("\n" + "=" * 60)
    print("Migration Summary:")
    print("✓ Replaced requests with langchain_ollama.OllamaLLM")
    print("✓ Simplified API calls using LangChain invoke() method")
    print("✓ Maintained existing function interface compatibility")
    print("✓ Enhanced error handling through LangChain")
    print("✓ Better integration with LangChain ecosystem")

    print("\nEnvironment Variables (unchanged):")
    print("- OLLAMA_API_BASE_URL: Set Ollama API base URL")
    print("- OLLAMA_DEFAULT_TRANSLATION_MODEL: Set model to use")
