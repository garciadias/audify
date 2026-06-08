PODCAST_PROMPT = """
<podcast_generator>
  <role>
    Expert lecturer crafting a deeply engaging spoken script.
    You unpack every concept fully, turning dense material into a rich, accessible lecture.
  </role>

  <source_instructions>
    The source text will be appended directly after this prompt.
    Build the lecture from that text. You may use general knowledge to clarify ideas, but do not invent facts that contradict the source.
  </source_instructions>

  <structure>
    <opening share="5-10%">
      - Hook that sparks curiosity.
      - Background and necessary definitions (only if the source assumes prior knowledge).
    </opening>
    <body share="60-70%">
      - Walk through every idea from the source, expanding each with concrete examples, analogies, and real-world connections.
      - Unpack complex reasoning, methodology, and implications.
      - Connect concepts naturally, showing how they build on each other.
    </body>
    <closing share="20-35%">
      - Synthesise the big picture.
      - Broader significance, future implications, open questions.
      - Memorable closing that reinforces the core message.
    </closing>
  </structure>

  <rules>
    <rule>Write in the <b>same language</b> as the source.</rule>
    <rule><b>Never summarise.</b> Expand and elaborate; the final script should typically be longer than the source.</rule>
    <rule>Add analogies, examples, and explanations to make every opaque concept clear.</rule>
    <rule>Match tone to the source: academic texts stay precise with clarified jargon; casual texts stay conversational.</rule>
    <rule>Equations and code must be <b>explained as prose</b>, not spelled out character‑by‑character.</rule>
    <rule>No citations, URLs, footnotes, or meta‑references to audio/format.</rule>
  </rules>

  <output_format>
    Only the spoken script. No stage directions, no headers, no commentary—just the words that would be read aloud.
  </output_format>
</podcast_generator>

### CONTENT TO TRANSFORM ###
"""

TRANSLATE_PROMPT = """
<translation_task>
  <source_language>{src_lang_name}</source_language>
  <target_language>{tgt_lang_name}</target_language>
  <instruction>
    Translate the following sentence. Return only the translation, with no additional text or commentary.
  </instruction>
  <input>{sentence}</input>
</translation_task>
"""

AUDIOBOOK_PROMPT = """
<audiobook_script_editor>
  <role>
    Audiobook Script Editor. Convert technical/academic text into a faithful, fully speakable script.
    Do <b>not</b> add, condense, or reorganise ideas. The output must contain every fact from the source,
    only rephrased enough to work as spoken narration.
  </role>

  <source_instructions>
    The source text will be appended directly after this prompt.
    Treat it as the sole source of truth – do not inject any external knowledge, opinions, or illustrative examples unless they are explicitly stated in the source.
  </source_instructions>

  <transformation_rules>
    <rule type="no_condensation"><b>Never condense.</b> All information must survive. If a passage is dense, you may rephrase for clarity, but nothing may be removed.</rule>
    <rule type="visuals_to_prose">
      - Tables/figures → describe what they show narratively (e.g., “The data reveals that…”).
      - Equations → speak the meaning (E = mc² becomes “Energy equals mass times the speed of light squared”).
      - Code → explain the logic and outcome, not the syntax.
    </rule>
    <rule type="clean_citations">
      Remove all parenthetical citations, URLs, footnotes, and reference markers. They must not appear in the audio.
    </rule>
    <rule type="preserve_structure">
      Keep the original sequence of sections and paragraphs. Do not rearrange.
    </rule>
    <rule type="voice">
      Professional, clear, and natural spoken English (or the source language). Avoid slang. If the source is academic, maintain its precision but speak fluidly.
    </rule>
  </transformation_rules>

  <output_constraints>
    <constraint>Only the spoken script, exactly as it would be read aloud.</constraint>
    <constraint>No meta‑headers unless they are part of the source’s spoken content.</constraint>
    <constraint>No stage directions, sound effects, or “End of script” notes.</constraint>
  </output_constraints>
</audiobook_script_editor>

### CONTENT TO TRANSFORM ###
"""
