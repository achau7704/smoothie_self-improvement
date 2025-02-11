"""
Utility functions for self-improvement v2
"""

def normalize_pythia(generation: str) -> str:
    """
    Normalize a string from a Pythia-like model output by:
      1) Converting to lowercase.
      2) Extracting everything after 'A:'.
      3) Stripping leading/trailing whitespace.
    """
    # 1) Convert to lowercase
    text_lower = generation.lower()

    # 2) Split into lines, filter out lines with "q:"
    lines = text_lower.splitlines()
    filtered = [line for line in lines if "q:" not in line]

    # 3) Find first line containing "a:", keep only what's after "a:",
    #    and discard subsequent lines
    final_answer = ""
    for line in filtered:
        if "a:" in line:
            # Extract everything after "a:"
            final_answer = line.split("a:", 1)[1]
            break
        # Otherwise, ignore lines before we see "a:" and do not keep them

    # 4) Strip whitespace
    final_answer = final_answer.strip()

    return final_answer
