PODCAST_PROMPT = """
Transform the following content into a comprehensive, detailed
talk that thoroughly explores every aspect of the material.
This will be converted directly to speech, so write in a natural, conversational tone
suitable for audio consumption.

CRITICAL REQUIREMENTS:
- Create an EXTENSIVE, VERBOSE explanation covering ALL details from the source material
- Aim for at least the same amount of text as the original content
- Do NOT summarize - EXPAND and ELABORATE on every concept, idea, and detail
- Use a engaging lecture style as if explaining to an intelligent but curious audience

STRUCTURE YOUR RESPONSE AS FOLLOWS:

1. (5-10% of total content):
   - Begin with an engaging hook that draws listeners in
   - Provide extensive background context and historical perspective
   - Explain ALL prerequisite knowledge needed to understand the material
   - Define technical terms, concepts, and theories in detail
   - Discuss the broader significance and relevance of the topic
   - Set up the context for why this content matters today

2. (60-70% of total content):
   - Go through the material systematically and thoroughly
   - Elaborate on EVERY important point, concept, and finding
   - Provide multiple examples and analogies to clarify complex ideas
   - Discuss the methodology, reasoning, or approach behind key points
   - Analyze the implications and connections between different concepts
   - Include relevant real-world applications and examples
   - Discuss any controversies, debates, or alternative perspectives
   - Use transition phrases to maintain flow between topics

3. (20-35% of total content):
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
- DO NOT include any text that would not be read aloud
- DO NOT include any of these instructions in the output
The goal is to create rich, comprehensive audio content that thoroughly educates and
engages listeners with detailed explanations of every aspect of the source material.

Content to transform:
--------------------
"""
TRANSLATE_PROMPT = """
Translate the following text from {src_lang_name} to {tgt_lang_name}.
Only return the translated text, nothing else.

Text to translate: {sentence}

Translation:"""
AUDIOBOOK_PROMPT = """
You are the editor of a audiobook platform. Your main job is to adapt books to the
audiobook format. You will receive a chapter of a book and you MUST apply the following
instructions to that.

Clean the following content to better adapt to the audio format.

CRITICAL REQUIREMENTS:
- Aim for preserving the most of the original content
- Do NOT summarize - EXPAND and ELABORATE on key concept, ideas, and details that you
think are relevant.
- Remove citations, references, length code blocks, equations and tables. Since these
be disruptive for the reading. Process the content of these and explain their mean.
- Write in the same language as the source text
- Use concrete examples and analogies whenever possible
- NO references, citations, bibliographies, or URL mentions
- NO stage directions or meta-commentary about the content format
- ONLY include content that should be spoken aloud
- When you find the first reference to a figure, table or code block or equation, you
must describe it by using the captions of that item or extracting relevant information
from the code or equation.

The goal is to create rich, comprehensive audio content that thoroughly educates and
engages listeners with detailed explanations of every aspect of the source material.

Chapter content to transform:
-----------------------------

"""
