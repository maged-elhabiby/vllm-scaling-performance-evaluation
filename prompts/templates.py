"""
Prompt templates and padding strategies for synthetic workload generation

Prompts are built from a realistic instruction-following template padded with
a filler passage so that the total input token count matches the target exactly.
Using a natural-language filler (rather than repeated single tokens) avoids
tokenizer edge cases and produces KV-cache access patterns representative of
real workloads.
"""

# Base instruction that anchors every prompt regardless of length, kept short so the bulk of the token budget is filled by the padding passage
BASE_INSTRUCTION = (
    "You are a helpful assistant. Read the following background text carefully "
    "and then answer the question at the end.\n\n"
    "Background text:\n"
)

QUESTION_SUFFIX = (
    "\n\nBased on the background text above, write a brief summary."
)

# A natural-language passage used to pad prompts to the desired token count
# TODO: CHANGE AND VALIDATE, GRABBED RANDOMLY ATM
PADDING_PASSAGE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
    "Bright vixens jump dozing fowl quack. "
    "Sphinx of black quartz judge my vow. "
    "Two driven jocks help fax my big quiz. "
    "Five quacking zephyrs jolt my wax bed. "
    "The jay pig fox zebra and my wolves quack. "
    "Blowzy night frumps vex a dq jock. "
)

# Repeat the passage enough times that we never run out of padding material
PADDING_PASSAGE_LONG = PADDING_PASSAGE * 200
