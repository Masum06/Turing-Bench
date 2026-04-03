#  MULTITHREADING CONFIG
USE_THREADS = True   # Set True to enable parallel inference
N_THREADS   = 4       # Number of worker threads (ignored if USE_THREADS = False)

MAX_RETRIES = 5
BASE_DELAY  = 1.0   # seconds — doubles each attempt: 1, 2, 4, 8, 16

#  PROMPTS
SYSTEM_PROMPT = """\
INSTRUCTIONS:
The following a task is an offline Turing test where you will be presented with two dialogues and your job is to identify the human.

In this experiment, you will be presented with a
pair of conversations, labeled "Transcript A" and "Transcript B", each one between an investigator I and a witness W. 
One of them is a human/human conversation and another is a human/AI conversation. Ignore all statements or actions by investigator I, 
and only focus on the witness W. Your job will be to decide in which conversation (A or B)
the Witness W is human. The AI (LLM or otherwise) is highly advanced, so the differences may be very subtle, 
hence pay close attention.

Each conversation involves:
A Witness, labeled 'W'
An Investigator, labeled 'I'

The Investigator (I) is always human. In exactly one
of the conversations, the Witness W is human, and
in the other, the Witness W is Artificial Intelligence.

Please provide your response in JSON format:

{
"result": {
  "verdict": <"A" or "B">,
  }
}
"""

USER_TEMPLATE = """
=== Transcript A ===
{dialogueA}

=== Transcript B ===
{dialogueB}

Output:
"""