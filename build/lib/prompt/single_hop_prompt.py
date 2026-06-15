SINGLE_HOP_PROMPT = """\
You are an expert dataset curator building a high-quality RAG fine-tuning dataset from scientific papers.
 
Your task is to generate {n} questions from the chunk below. Each question must require genuine reasoning to answer — not copying or locating a phrase.
 
Paper Title: {title}
Section: {section}
 
--- CHUNK START ---
{chunk}
--- CHUNK END ---
 
=== STEP 1: INTERNAL ANALYSIS (do this reasoning before writing any question) ===
 
Before generating questions, identify the following FROM THIS CHUNK ONLY:
- What is the main claim, mechanism, or finding described?
- What causal relationships or design decisions are explained?
- Are there any comparisons, trade-offs, or contrasts stated?
- What motivations or implications are explicitly given?
 
IMPORTANT: Only use what is explicitly in the chunk. Do not infer what is not stated.
 
=== STEP 2: QUESTION GENERATION RULES ===
 
REQUIRED — Each question MUST:
1. Be answerable using ONLY this chunk (no external knowledge).
2. Require the reader to synthesize, explain, or reason — NOT locate a phrase.
3. Have an answer spanning multiple sentences or ideas from the chunk.
4. Be specific to the actual content of this chunk (not generic enough to apply to any paper).
 
FORBIDDEN — Reject any question that:
- Can be answered by copying a single sentence or phrase from the chunk.
- Asks for names, lists, or identifiers (e.g., "What dataset?", "What is the model called?").
- Uses the word "mention" or "describe" in a way that invites listing.
- Would be valid as a question on ANY paper, not specifically this chunk.
- Requires knowledge from outside this chunk to answer fully.
 
REASONING DEPTH TEST — Before finalizing each question, ask yourself:
"Can a reader answer this by scanning for a keyword and copying the nearby sentence?"
→ If YES: discard and generate a harder question.
→ If NO: keep it.
 
ACCEPTED QUESTION TYPES (only if grounded in this chunk):
- Explain the mechanism: "How does X achieve Y according to this section?"
- Explain a workflow: "What steps does the process follow and why is each step necessary?"
- Interpret a result: "What does the reported [metric/outcome] indicate about [aspect]?"
- Contrast: "How does [A] differ from [B] as described here?" (only if both are in the chunk)
- Causal: "Why does the method use [specific design] according to this section?" (only if reason is stated)
- Trade-off: "What does the paper identify as the cost of [design choice]?" (only if stated)
 
=== STEP 3: OUTPUT ===
 
Return ONLY a valid JSON array of question strings.
No preamble, no explanation, no markdown.
 
[
  "question 1",
  "question 2"
]
"""