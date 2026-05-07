#  MULTITHREADING CONFIG
USE_THREADS = True   # Set True to enable parallel inference
N_THREADS   = 10      # Number of worker threads (ignored if USE_THREADS = False)

MAX_RETRIES = 5
BASE_DELAY  = 1.0   # seconds — doubles each attempt: 1, 2, 4, 8, 16

#  PROMPTS
SYSTEM_PROMPT = open("prompt.txt", "r").read()

USER_TEMPLATE = """
=== Transcript A ===
{dialogueA}

=== Transcript B ===
{dialogueB}

Output:
"""