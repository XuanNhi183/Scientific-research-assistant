MULTI_HOP_PROMPT = """\
You are an expert dataset curator building a high-quality RAG fine-tuning dataset from scientific papers.
 
Your task is to generate {n} multi-hop questions that are ONLY answerable by combining specific information from BOTH chunks below.
 
Paper Title: {title}
 
--- CHUNK A START ---
Section: {section_a}
{chunk_a}
--- CHUNK A END ---
 
--- CHUNK B START ---
Section: {section_b}
{chunk_b}
--- CHUNK B END ---
 
=== STEP 1: INTERNAL DEPENDENCY ANALYSIS ===
 
Before writing any question, explicitly identify:
 
FROM CHUNK A ONLY — What specific fact, mechanism, or claim does Chunk A contribute?
FROM CHUNK B ONLY — What specific fact, mechanism, or claim does Chunk B contribute?
BRIDGE — What is the logical connection between these two pieces of information?
 
A valid multi-hop question MUST require BOTH of these specific contributions.
If the bridge is weak or generic (e.g., "both are about the model"), do NOT generate a question — return an empty array instead.
 
=== STEP 2: MANDATORY DEPENDENCY TEST ===
 
For every candidate question, apply this test before including it:
 
TEST A — Remove Chunk A. Can the question still be fully answered using only Chunk B?
→ If YES: the question is INVALID. Discard it.
 
TEST B — Remove Chunk B. Can the question still be fully answered using only Chunk A?
→ If YES: the question is INVALID. Discard it.
 
Only questions that FAIL both tests (i.e., require both chunks) are valid.
 
=== STEP 3: QUESTION QUALITY RULES ===
 
REQUIRED:
1. The question must name or reference something specific from each chunk — not generic connectors like "the method" or "the results".
2. The answer must require synthesis: combining two distinct pieces of information into a coherent response.
3. The question must be naturally motivated — a real researcher would actually ask this.
4. The question must be answerable ONLY from these two chunks with no external knowledge.
 
FORBIDDEN:
- Questions where one chunk is just "context" and the other has the real answer.
- Questions phrased so broadly that Chunk A or B alone gives a sufficient answer.
- Questions that use generic academic language to disguise single-chunk answerability (e.g., "How do the results validate the method?" — almost always answerable from results chunk alone).
- Template-filling: do NOT map patterns like "Method + Result" onto the chunks if no genuine dependency exists.
 
SIGNAL OF A BAD QUESTION:
→ The question reads like it could appear in any paper about any method.
→ The answer from one chunk is complete; the other chunk only adds minor detail.
 
SIGNAL OF A GOOD QUESTION:
→ The question names a specific mechanism from Chunk A and asks how it connects to a specific observation in Chunk B (or vice versa).
→ Removing either chunk leaves the answer genuinely incomplete.
 
=== STEP 4: OUTPUT ===
 
If no valid multi-hop question can be constructed from these chunks, return an empty array: []
 
Otherwise return ONLY a valid JSON array of question strings.
No preamble, no explanation, no markdown.
 
[
  "question 1"
]
"""