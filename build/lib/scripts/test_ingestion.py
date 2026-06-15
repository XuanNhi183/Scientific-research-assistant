import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from service.chunking import extract_with_marker, init_marker_models, chunk_markdown

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Test PDF extraction and chunking with Marker.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to test.")
    parser.add_argument("--output", type=str, default="test_extraction_output.md", help="Path to save the extracted markdown.")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' does not exist.")
        sys.exit(1)

    print("Initializing Marker models... (this may take a moment)")
    converter = init_marker_models()

    print(f"\nExtracting text from: {pdf_path}")
    try:
        md_text = extract_with_marker(pdf_path, converter)
        print("Extraction successful!")
    except Exception as e:
        print(f"Failed to extract with Marker: {e}")
        sys.exit(1)

    # Save the full markdown output
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Saved full Markdown extraction to: {args.output}")

    print("\nRunning chunking on the extracted markdown...")
    chunks = chunk_markdown(md_text)
    print(f"Successfully split into {len(chunks)} chunks.")

    print("\n--- Preview of first 2 chunks ---")
    for i, c in enumerate(chunks[:2]):
        print(f"\n[Chunk {i+1}] - Section: {c.metadata.section}")
        preview_text = c.text[:300] + ("..." if len(c.text) > 300 else "")
        print(preview_text)
        print("-" * 50)

if __name__ == "__main__":
    main()