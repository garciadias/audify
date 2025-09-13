from pathlib import Path

LANG_CODES = {
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "pt": "p",
    "en": "a",
    "zh": "z",
    "ja": "j",
}

DEFAULT_LANGUAGE_LIST = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh",
    "hu",
    "ko",
    "ja",
    "hi",
]
KOKORO_DEFAULT_VOICE = "af_bella"

MODULE_PATH = Path(__file__).parent.resolve()
OUTPUT_BASE_DIR = MODULE_PATH / "data" / "output"

PODCAST_INSTRUCTIONS = """Transform the following content into a comprehensive, detailed
talk that thoroughly explores every aspect of the material.
This will be converted directly to speech, so write in a natural, conversational tone
suitable for audio consumption.

CRITICAL REQUIREMENTS:
- Create an EXTENSIVE, VERBOSE explanation covering ALL details from the source material
- Aim for at least the same amount of text as the original content
- Do NOT summarize - EXPAND and ELABORATE on every concept, idea, and detail
- Use a engaging lecture style as if explaining to an intelligent but curious audience

STRUCTURE YOUR RESPONSE AS FOLLOWS:

1. COMPREHENSIVE INTRODUCTION (5-10% of total content):
   - Begin with an engaging hook that draws listeners in
   - Provide extensive background context and historical perspective
   - Explain ALL prerequisite knowledge needed to understand the material
   - Define technical terms, concepts, and theories in detail
   - Discuss the broader significance and relevance of the topic
   - Set up the context for why this content matters today

2. DETAILED MAIN CONTENT EXPLORATION (60-70% of total content):
   - Go through the material systematically and thoroughly
   - Elaborate on EVERY important point, concept, and finding
   - Provide multiple examples and analogies to clarify complex ideas
   - Discuss the methodology, reasoning, or approach behind key points
   - Analyze the implications and connections between different concepts
   - Include relevant real-world applications and examples
   - Discuss any controversies, debates, or alternative perspectives
   - Use transition phrases to maintain flow between topics

3. COMPREHENSIVE CONCLUSION (20-35% of total content):
   - Synthesize all the key insights and takeaways
   - Discuss the broader implications and future directions
   - Highlight what makes this content particularly significant or innovative
   - Suggest questions for further reflection
   - Emphasize the main milestones covered

IMPORTANT GUIDELINES:
- Write in the same language as the source text
- Write as if you're an expert lecturer who is passionate about the subject
- Include phrases like "Now, let's dive deeper into..." or "This is particularly
  fascinating because..."
- Provide context for why each point matters
- Use concrete examples and analogies whenever possible
- Maintain an enthusiastic but informative tone throughout
- NO references, citations, bibliographies, or URL mentions
- NO stage directions or meta-commentary about the podcast format
- NO descriptions like "music fading" or anything unrelated to spoken content
- DO NOT mention the podcast
- NO music
- NO directions for sound effects or podcast recording
- ONLY include content that should be spoken aloud

The goal is to create rich, comprehensive audio content that thoroughly educates and
engages listeners with detailed explanations of every aspect of the source material.

Content to transform:
--------------------
"""
DEFAULT_SPEAKER = KOKORO_DEFAULT_VOICE
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_ENGINE = "kokoro"
