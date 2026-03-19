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
# TODO: CHANGE AND VALIDATE, GRABBED RANDOMLY ATM - CHANGED
# Natural-language passages from Wikipeedia matching realistic LLM workloads.
# Produced varied, non-repetitive token sequences that stress the KV-cache.
# At 200 repititions - 120, 000 tokens - we have more than enough padding material to reach 100k tokens in the final prompt.

PADDING_PASSAGE = (
    "The central processing unit is the electronic circuitry that executes "
    "instructions comprising a computer program. The CPU performs basic "
    "arithmetic, logic, controlling, and input/output operations specified by "
    "the instructions in the program. This contrasts with external components "
    "such as main memory and I/O circuitry, and specialized coprocessors such "
    "as graphics processing units. "
 
    "A computer's memory hierarchy takes advantage of the principle of locality "
    "of reference by using smaller and faster memory technologies close to the "
    "processor. Caches exploit temporal locality by keeping recently accessed "
    "data close to the CPU, and spatial locality by loading contiguous blocks "
    "of memory into cache lines when any word within them is accessed. "
 
    "The operating system is system software that manages computer hardware and "
    "software resources, and provides common services for computer programs. "
    "Time-sharing operating systems schedule tasks for efficient use of the "
    "system and may also include accounting software for cost allocation of "
    "processor time, mass storage, printing, and other resources. "
 
    "Distributed computing is a field of computer science that studies "
    "distributed systems, defined as systems in which several components located "
    "on networked computers communicate and coordinate their actions by passing "
    "messages to one another. The components interact with one another in order "
    "to achieve a common goal. Three significant challenges of distributed "
    "systems are maintaining concurrency of components, overcoming the lack of a "
    "global clock, and managing the independent failure of components. "
 
    "In computing, a cache is a hardware or software component that stores data "
    "so that future requests for that data can be served faster. The data stored "
    "in a cache might be the result of an earlier computation or a copy of data "
    "stored elsewhere. A cache hit occurs when the requested data can be found "
    "in a cache, while a cache miss occurs when it cannot. Cache hits are served "
    "by reading data from the cache, which is faster than recomputing a result "
    "or reading from a slower data store. "
 
    "Network throughput is the rate of successful message delivery over a "
    "communication channel. Data delivered may be carried over a physical or "
    "logical link, or can pass through a certain network node. Throughput is "
    "usually measured in bits per second, and sometimes in data packets per "
    "second or data packets per time slot. The system throughput or aggregate "
    "throughput is the sum of the data rates delivered to all terminals in a "
    "network. Throughput is related to, but distinct from, bandwidth and latency. "
 
    "Concurrency control in database management systems ensures that correct "
    "results for concurrent operations are generated, while getting those results "
    "as quickly as possible. Computer storage devices, including memory and "
    "disks, are shared resources. Uncontrolled access to shared resources can "
    "result in data corruption. Concurrency control provides solutions to this "
    "problem. Read and write operations on shared data must be carefully managed "
    "to prevent access conflicts. "
 
    "Transformer models have become the dominant architecture for natural "
    "language processing tasks. The attention mechanism at their core allows "
    "each token in a sequence to attend to all other tokens, capturing long-range "
    "dependencies that earlier recurrent architectures struggled with. During "
    "inference, the key and value projections for previously processed tokens are "
    "stored in a structure called the KV-cache, allowing the model to generate "
    "new tokens without recomputing attention over the full history. As sequence "
    "lengths grow, the KV-cache consumes an increasing share of GPU memory, "
    "creating a fundamental tension between context length and serving throughput. "
)

# Repeat the passage enough times that we never run out of padding material
PADDING_PASSAGE_LONG = PADDING_PASSAGE * 200
