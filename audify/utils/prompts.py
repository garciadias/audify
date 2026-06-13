# ruff: noqa: E501 - long lines are intentional in prompt templates
# PODCAST_PROMPT lives in audify/prompts/builtin/podcast.txt


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
