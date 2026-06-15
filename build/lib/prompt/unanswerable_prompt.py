UNANSWERABLE_PROMPT = """\
You are an expert dataset curator building a high-quality RAG fine-tuning dataset from scientific papers.

Your task is to generate {n} "unanswerable" questions based on the provided distractor chunk.
These questions must SOUND highly relevant to the chunk, but ask for a specific detail that is explicitly MISSING from the chunk.

Paper Title: {title}
Section: {section}

--- CHUNK START ---
{chunk}
--- CHUNK END ---

=== STEP 1: CHUNK ANALYSIS ===
Identify what the chunk is about. Is it an introduction? Methodology? Evaluation?
What specific entities, models, algorithms, or concepts are mentioned?

=== STEP 2: IDENTIFY MISSING DETAILS ===
Think of a detail that is strongly related to the entities in the chunk but is NOT mentioned.
For example:
- If the chunk mentions a model's architecture, ask about its exact hyperparameter values or training time (if missing).
- If the chunk mentions an experiment, ask about a baseline method or dataset that is NOT listed.
- If the chunk introduces a concept, ask about its limitations or specific mathematical formulation (if missing).

=== STEP 3: QUESTION GENERATION RULES ===
REQUIRED:
1. The question MUST use terminology, concepts, or names explicitly present in the chunk.
2. The question MUST ask for information that cannot be answered using the chunk.
3. The question should look like a genuine user query (e.g., "What was the learning rate used for the proposed model?").

FORBIDDEN:
- Do NOT generate questions that are completely unrelated to the chunk (e.g., asking about computer vision when the chunk is about NLP).
- Do NOT generate a question if the answer IS actually in the chunk. Double-check carefully!

=== OUTPUT FORMAT ===
Return ONLY a valid JSON array of question strings. No preamble, no markdown.

[
  "question 1"
]
"""
