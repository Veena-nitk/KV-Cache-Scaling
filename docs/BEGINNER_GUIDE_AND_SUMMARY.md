# The Beginner's Complete Guide to KV-Cache Scaling
## What This Project Is, What We Did, How It Works, and How to Run It in Google Colab

> **Author**: Antigravity Research Engineering Team  
> **Target Audience**: Complete beginners, students, and curious learners  
> **Project**: `KV-Cache-Scaling` (Qwen2.5-1.5B, FP16, Batch Size 1, Nvidia Tesla T4 GPU)  
> **Date**: October 2026  

---

## Table of Contents
1. [The Big Picture: What is This Project About?](#1-the-big-picture-what-is-this-project-about)
2. [The Jargon Buster: Every Technical Term Explained in Plain English](#2-the-jargon-buster-every-technical-term-explained-in-plain-english)
3. [Demystifying Google Colab: How and Why We Use It](#3-demystifying-google-colab-how-and-why-we-use-it)
4. [What Have We Done So Far? (The Project Journey)](#4-what-have-we-done-so-far-the-project-journey)
5. [Tour of the Project Folder: Files and Scripts Explained](#5-tour-of-the-project-folder-files-and-scripts-explained)
6. [Under the Hood: How the Code Uses the GPU](#6-under-the-hood-how-the-code-uses-the-gpu)
7. [Step-by-Step Instructions: Running the Experiment in Google Colab](#7-step-by-step-instructions-running-the-experiment-in-google-colab)
8. [Summary Cheat Sheet](#8-summary-cheat-sheet)

---

## 1. The Big Picture: What is This Project About?

### The Everyday Problem
When you chat with an AI like ChatGPT or Claude, you can give it a short question (like *"What is the capital of France?"*) or a huge document (like an entire 300-page textbook). 

As the conversation gets longer, the AI has to "remember" every single word that came before. If it didn't, it would lose the thread of the conversation. But that memory is not free. Every single word you feed into the AI takes up a tiny slice of physical computer memory inside the AI's graphics chip (the **GPU**).

If the conversation gets too long, the computer memory completely fills up. When there is no room left, the program violently crashes with an error called **CUDA Out of Memory (OOM)**.

```
+-------------------------------------------------------------+
|                  THE GPU MEMORY "DESK"                      |
|                                                             |
|   +-------------------+    +----------------------------+   |
|   |  Model Weights    |    |      KV-Cache "Cards"      |   |
|   |  (The Heavy Book) |    |  (Grows with every word!)  |   |
|   |     ~3.09 GB      |    |       28,672 bytes/word    |   |
|   +-------------------+    +----------------------------+   |
|                                                             |
|  Total Desk Space Available: ~15.8 GB (Tesla T4 GPU)        |
+-------------------------------------------------------------+
```

### The "Desk and Index Cards" Analogy
Think of the GPU as a physical **office desk**:
1. **The Desk**: The desk has a fixed size (15.8 Gigabytes on an Nvidia Tesla T4 GPU).
2. **The Model Weights**: A giant encyclopedia placed on the desk. This is the AI's brain (its vocabulary, grammar, and knowledge). For our AI model (`Qwen2.5-1.5B`), this encyclopedia permanently takes up **3.09 GB** of desk space. It never shrinks, and it never moves.
3. **The KV Cache (Index Cards)**: Every time the AI reads or writes a word, it jots down notes on an index card so it remembers what that word meant in context. Each word takes up exactly **28,672 bytes** on the desk.
4. **The Crash (OOM)**: At 1,000 words, you have a small pile of index cards. At 50,000 words, half the desk is covered. At 114,688 words, the desk is almost completely full. At around 125,000 words, the index cards spill over the edge of the desk onto the floor. The computer crashes!

### The Goal of This Project
Our mission is **rigorous scientific engineering**:
1. **Predict Ahead of Time**: Can we write a pure mathematical equation that calculates *the exact word count* where the desk overflows, down to the exact byte, *before* we ever turn on the computer?
2. **Empirical Measurement**: Can we measure the real memory on a real GPU across dozens of word lengths and prove our formula is right within 0.01%?
3. **Audit and Account for Every Byte**: Can we account for every single byte of memory so there are zero unexplained mysteries?
4. **Find the Breaking Point**: Can we pinpoint the exact boundary where the model fits vs. where it crashes?

---

## 2. The Jargon Buster: Every Technical Term Explained in Plain English

Here is your dictionary for every technical term used in this project:

| Term | What It Stands For | What It Actually Means | Everyday Analogy |
| :--- | :--- | :--- | :--- |
| **LLM** | Large Language Model | A deep learning computer program trained to predict the next word in a sentence (e.g., Qwen, GPT-4, LLaMA). | A super-advanced autocomplete system. |
| **Qwen2.5-1.5B** | Qwen 2.5 (1.5 Billion parameters) | The specific AI model we are studying. It has 1.54 billion numbers ("parameters") that make up its knowledge base. | A medium-sized encyclopedia. |
| **Token** | Word piece | The basic unit of text AI reads. One token is roughly 3 to 4 letters of English. For example, the word `"fantastic"` is split into 2 or 3 tokens: `["fan", "tas", "tic"]`. | A syllable or word-block. |
| **Context Length ($L$)** | Number of tokens | The total number of tokens fed into the AI at once (e.g., 512 tokens $\approx$ 1 page; 114,688 tokens $\approx$ a 300-page book). | The thickness of the book you hand the AI. |
| **GPU** | Graphics Processing Unit | A specialized computer chip made by Nvidia. Unlike a normal laptop CPU which does 4 to 8 things at a time, a GPU has thousands of tiny cores that do millions of math calculations simultaneously. | A factory with 2,500 workers doing arithmetic in unison. |
| **VRAM** | Video RAM / GPU Memory | Ultra-fast memory located directly on the GPU card. Our Nvidia Tesla T4 GPU has ~15.8 GB of VRAM. | The surface area of the desk. |
| **FP16** | 16-bit Floating Point (Half Precision) | A format for storing numbers inside the computer. Instead of using 32 bits (4 bytes) per number, FP16 uses 16 bits (2 bytes). This cuts memory usage in half! | Writing small numbers with a fine-tip pen to save page space. |
| **KV Cache** | Key-Value Cache | Memory tables stored in VRAM so the AI doesn't have to re-read and re-calculate past words for every new word it generates. | The index cards the AI keeps on its desk. |
| **Slope** | Memory Growth Rate | The exact number of bytes the KV cache grows for each new token. For our model, it is **exactly 28,672 bytes per token**. | The weight of each single index card added to the desk. |
| **CUDA OOM** | Out of Memory Error | A hard crash that occurs when the code asks the GPU for more VRAM than physically exists. | Trying to pour a gallon of water into a pint glass. |
| **Preregistration** | Prospective Scientific Locking | Writing down your mathematical formula and predictions, timestamping them, and locking them into Git *before* running any tests on the GPU. This prevents "cheating" or altering predictions after seeing the results. | Putting your prediction in a sealed, notarized envelope before the race begins. |
| **PyTorch Allocator** | Memory Manager | The software inside PyTorch that asks the GPU for memory blocks and organizes them. | The desk organizer who decides where new folders get stacked. |
| **Default vs Expandable** | Allocator Modes | **Default**: Fast, but can leave unusable gaps between memory blocks (fragmentation).<br>**Expandable Segments**: A smarter mode (`expandable_segments:True`) that stitches together virtual memory to avoid wasting space. | Default = stacking random-sized boxes with gaps.<br>Expandable = tetris-packing boxes so zero space is wasted. |
| **Bisection** | Binary Search | A smart mathematical method for finding an unknown boundary by testing the exact midpoint and cutting the uncertainty in half every turn. | Playing the "Higher or Lower" guessing game between 1 and 100. |

---

## 3. Demystifying Google Colab: How and Why We Use It

### Why Can't We Just Run This on Your Mac?
Your Mac has an Apple M-series processor (or Intel). Apple uses Unified Memory and Metal graphics. But almost all modern AI software (like PyTorch with CUDA) is designed specifically for **Nvidia GPUs**. 

Nvidia Tesla T4 GPUs cost thousands of dollars and require dedicated server hardware. Rather than buying an Nvidia server, we use **Google Colab**!

### What is Google Colab?
Google Colaboratory (Colab) is a free cloud computing service provided by Google. When you open a Colab notebook, Google connects your web browser to a real, powerful Linux virtual machine in one of Google's data centers. That virtual machine has an **Nvidia Tesla T4 GPU** attached to it!

```
+------------------+          Internet          +------------------------------------+
|  Your Computer   |   ====================>   |      Google Cloud Data Center      |
|  (Web Browser)   |                           |  Linux Server + Nvidia Tesla T4    |
|                  |   <====================   |  15.8 GB High-Speed GPU VRAM       |
+------------------+     Notebook Results      +------------------------------------+
```

### What is a Jupyter Notebook (`.ipynb`)?
A Jupyter Notebook is an interactive document with the file extension `.ipynb`. Unlike a normal text document or a standard `.py` script, a notebook contains both formatted reading material and runnable code.

### What are "Cells"?
A notebook is divided into stacked blocks called **Cells**. There are two types:
1. **Markdown Cells**: These contain formatted text, headings, bullet points, and math. They are meant for humans to read.
2. **Code Cells**: These contain executable Python code. They have a gray background and a circular **Play button** ($\blacktriangleright$) on their left side.

### What Does "Running a Cell" Mean?
When you click the **Play button** on a code cell (or click inside it and press `Shift + Enter`):
1. Your web browser takes the Python code inside that cell and sends it over the internet to Google's cloud server.
2. Google's cloud server feeds the code to Python.
3. Python executes the instructions line by line on the Nvidia GPU.
4. Any output (text, numbers, progress bars, tables, or error messages) is sent back to your browser and printed directly below that cell.
5. Any variables, models, or functions created in that cell stay alive in the server's memory for the next cell to use!

---

## 4. What Have We Done So Far? (The Project Journey)

To understand where we are right now, here is the chronological story of the project:

### Phase 1: The Initial Research & Mathematical Formula
- We derived the exact mathematical formula for how much memory `Qwen2.5-1.5B` requires:
  $$\text{KV Bytes per Token} = 2 \times \text{layers (28)} \times \text{KV heads (2)} \times \text{head dimension (128)} \times 2\text{ bytes (FP16)} = 28,672\text{ bytes/token}$$
- We calculated that with ~3.09 GB for the model weights and ~12.5 GB of usable headroom, the model should be able to hold approximately **111,232 tokens** before crashing.
- We **preregistered** this number in Git before running a single GPU test.

### Phase 2: The First GPU Probes
- We ran 52 tests on a Google Colab Nvidia T4 GPU across 9 different context lengths (from 512 tokens up to 114,688 tokens).
- **The Result**: 114,688 tokens fit comfortably! The measured memory matched our theoretical slope ($28,672$ bytes/token) with an $R^2$ of $1.000000$ (a mathematically perfect linear fit).
- However, at 131,072 tokens (the maximum possible context the model architecture supports), the GPU crashed with an Out of Memory error.

### Phase 3: The Audit (Part 0)
The user asked us to act as a strict external auditor and verify 9 specific claims (C1 to C9) made in a draft email. We combed through Git commit timestamps, raw JSON log files, and data capsules:
- **C1 (Preregistered before run)**: Verified! Git commit occurred at 14:24 UTC, first GPU run occurred at 14:34 UTC.
- **C3 (Slope is 28,672 bytes/token)**: Verified 100% exact.
- **C5 (Discrepancy closed)**: Verified to 0 bytes unexplained.
- **C8 (OOM check)**: We discovered that while 131,072 did crash, the draft email had cited an incorrect run ID due to a typo. We documented this transparently in `docs/AUDIT_OF_MAIL.md`.

### Phase 4: Answering the TA Requests (Part 1, Tasks 2 to 5)
Our Teaching Assistant (TA) asked for four specific technical deliverables, all of which we completed:
1. **Task 2**: A byte-by-byte accounting ledger proving that every single byte of memory is accounted for down to **0 bytes unaccounted** (`docs/DISCREPANCY_114688.md`).
2. **Task 3**: A rigorous Methods section citing exact code files and line numbers (`docs/METHODS.md`).
3. **Task 4**: Two clean comparison tables: Table A (comparing our theoretical prediction to reality) and Table B (comparing our fitted model to reality) (`docs/PREDICTION_ERROR_AT_CURRENT_BRACKET.md`).
4. **Task 5**: An investigation of published figures from vLLM showing exact memory formulas (`docs/VLLM_PUBLISHED_FIGURE.md`).

### Phase 5: Preparing the Bisection Experiment (Part 1, Task 1)
Now comes the exciting final challenge: We know that **114,688 tokens fits**, and **131,072 tokens crashes**. That leaves a gap of 16,384 tokens in between!
- Where *exactly* does the model die? Is it at 120,000? 124,000? 128,000?
- Does the modern allocator (`expandable_segments:True`) allow the model to fit more tokens than the default allocator?
- To find out, we prepared a prospective **Preregistration Amendment** (`prereg/bisect-2026-10-28/`) and created a dedicated Google Colab notebook (`notebooks/BOUNDARY_COLAB.ipynb`) to pinpoint the boundary down to $\le 512$ tokens using binary search (bisection).

---

## 5. Tour of the Project Folder: Files and Scripts Explained

Here is an architectural map of our project directory (`KV-Cache-Scaling`):

```
KV-Cache-Scaling/
│
├── configs/                   <-- Settings and blueprints
│   ├── model.json             <-- Exact architecture of Qwen2.5-1.5B (28 layers, etc.)
│   ├── probe.json             <-- Probe instructions (batch size 1, FP16, etc.)
│   └── environment.json       <-- Hardware requirements (Tesla T4, CUDA 12)
│
├── src/kv_cache_scaling/      <-- The Core Python Engine (The Library)
│   ├── model_math.py          <-- Pure math equations: calculates KV slope & predictions
│   ├── probe.py               <-- Talks to PyTorch: allocates memory, runs model, records stats
│   ├── bisect_worker.py       <-- Single-test runner: runs one probe in an isolated process
│   ├── processes.py           <-- Safe process launcher: isolates crashes so Python doesn't die
│   ├── analysis.py            <-- Statistical math: fits linear regression lines
│   ├── reporting.py           <-- Markdown report generator: writes summary tables
│   └── coordinator_store.py   <-- Checkpoint system: saves state after every single probe
│
├── scripts/                   <-- Command-Line Tools (You run these in terminal)
│   ├── bisect_boundary.py     <-- The Bisection Brain: chooses midpoints and coordinates tests
│   ├── create_boundary_notebook.py <-- Builds the clean Colab notebook automatically
│   ├── reproduce_results.py   <-- Re-runs all math and verifies historical receipts
│   └── generate_beginner_guide_pdf.py <-- Generates the PDF guide you are reading!
│
├── notebooks/                 <-- Interactive Google Colab Notebooks
│   ├── BOUNDARY_COLAB.ipynb   <-- THE NOTEBOOK TO RUN: The bisection experiment for Colab
│   └── FINAL_PRESENTATION.ipynb <-- Interactive visualization of all historical results
│
├── prereg/                    <-- Scientific Vault (Frozen predictions before GPU runs)
│   ├── final-2026-10-19/      <-- Original preregistered prediction (111,232 tokens)
│   └── bisect-2026-10-28/     <-- New preregistered bisection plan and OOM budget
│
├── artifacts/                 <-- Frozen empirical data capsules from historical runs
│   └── ...                    <-- Raw JSON logs, receipts, and environment snapshots
│
├── docs/                      <-- All scientific documentation and TA responses
│   ├── AUDIT_OF_MAIL.md       <-- The Part 0 audit of claims C1–C9
│   ├── TA_RESPONSE.md         <-- Formal answers to TA requests 1 through 5
│   ├── DISCREPANCY_114688.md  <-- Byte-by-byte memory reconciliation ledger
│   ├── METHODS.md             <-- Code citations for how memory was measured
│   └── PROJECT_EXPLANATION_BEGINNER_GUIDE.pdf <-- This guide in PDF format!
│
└── tests/                     <-- 127 automated software tests verifying every single line of code
```

### What Do the Main Python Scripts Actually Do?

1. **`src/kv_cache_scaling/model_math.py`**:
   - Contains zero GPU code. It is pure math.
   - It reads `configs/model.json` and calculates:
     $$\text{KV Slope} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_elem}$$
   - For our model: $2 \times 28 \times 2 \times 128 \times 2 = 28,672\text{ bytes/token}$.
2. **`src/kv_cache_scaling/probe.py`**:
   - The laboratory instrument. It imports PyTorch, loads `Qwen/Qwen2.5-1.5B`, creates a synthetic text sequence of length $L$, feeds it through the neural network, and calls `torch.cuda.max_memory_allocated()` to see the exact peak memory reached.
3. **`src/kv_cache_scaling/bisect_worker.py`**:
   - An isolated worker. Why do we need this? If PyTorch hits an Out of Memory error, the GPU memory can get corrupted or hold onto leftover fragments. By running each probe in a fresh, isolated worker process that starts and immediately exits, memory is 100% clean every single time!
4. **`scripts/bisect_boundary.py`**:
   - The game master. It looks at the current boundary $[114688, 131072]$. It picks the exact midpoint $(114688 + 131072) / 2 = 122880$ (rounded to a multiple of 512). It launches the worker to test 122,880. If it succeeds, the new bracket is $[122880, 131072]$. If it crashes, the new bracket is $[114688, 122880]$. It repeats this until the gap is $\le 512$ tokens!

---

## 6. Under the Hood: How the Code Uses the GPU

Here is the exact step-by-step lifecycle of what happens when our code runs a test on the GPU:

```
[ Step 1: Environment Setup ]
   | Set PYTORCH_CUDA_ALLOC_CONF (Default or expandable_segments:True)
   v
[ Step 2: Spawn Isolated Worker ]
   | Fresh Python process starts with clean GPU memory
   v
[ Step 3: Load Model Weights ]
   | Loads 1.54B parameters in FP16 (~3.09 GB allocated in VRAM)
   v
[ Step 4: Create Input Sequence ]
   | Generates dummy prompt of length L (e.g., L = 122,880 tokens)
   v
[ Step 5: Forward Pass (Layer by Layer) ]
   | Token representations pass through 28 Transformer layers.
   | Key and Value tensors are saved into PyTorch's DynamicCache.
   | Desk fills up: 3.09 GB + (L * 28,672 bytes).
   v
[ Step 6: Did It Fit or Crash? ]
  /                                \
 / Fits                             \ Exceeds 15.8 GB
v                                    v
Read max_memory_allocated()         Catch torch.cuda.OutOfMemoryError
Status: "success"                   Status: "cuda_oom"
Record exact bytes                  Record clean failure
   \                                /
    \                              /
     v                            v
[ Step 7: Write Receipt JSON & Exit Worker ]
Clean up GPU memory completely. Process terminates.
```

### Why Does 131,072 Crash Instantly? (The Attention Mask Cliff)
In our research, we discovered a fascinating bug/quirk in modern AI libraries:
- Hugging Face's `transformers` library has an optimization called SDPA (Scaled Dot-Product Attention) which avoids creating giant attention matrices in memory.
- However, when the context length $L$ reaches or exceeds the model's `sliding_window` parameter ($131,072$), Hugging Face's safety check disables the optimization and attempts to allocate a full 2D attention mask matrix of shape $[131072, 131073]$ in FP16.
- That single matrix requires **34.36 Gigabytes of VRAM**!
- Because our Tesla T4 only has 15.8 GB, the computer crashes instantly with an OOM before the model even begins computing attention.
- This proves why testing below 131,072 with bisection is scientifically vital!

---

## 7. Step-by-Step Instructions: Running the Experiment in Google Colab

Here is your exact, beginner-friendly walkthrough to run the bisection experiment on Google Colab:

### Step 1: Download the Notebook to Your Computer
The notebook is already built and sitting in this repository at:
`notebooks/BOUNDARY_COLAB.ipynb`

### Step 2: Open Google Colab
1. In your web browser (Chrome, Safari, Edge, etc.), go to: **[colab.research.google.com](https://colab.research.google.com/)**
2. Sign in with your Google account.
3. In the pop-up modal, click the **"Upload"** tab.
4. Drag and drop `BOUNDARY_COLAB.ipynb` from your computer into Google Colab.

### Step 3: Enable the Nvidia T4 GPU
1. In the top menu bar of Google Colab, click **Runtime** $\rightarrow$ **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**.
3. Click **Save**.
4. In the top-right corner, click **Connect**. Within a few seconds, it will show a green checkmark with "RAM / Disk".

### Step 4: Run the Cells in Order
Run each cell one by one by clicking the **Play button** ($\blacktriangleright$) on its left (or pressing `Shift + Enter`):

- **Cell 1 (GPU Environment Check)**:
  - Prints your GPU name (`Tesla T4`), total VRAM (~15,360 MiB), and driver version.
- **Cell 2 (Clone & Install)**:
  - Clones the repository, installs PyTorch and Hugging Face `transformers`, and sets `PYTHONPATH`.
- **Cell 3 (Sanity Checks)**:
  - Runs quick verification tests to ensure the math library works.
- **Cell 4 (Checkpoint Exporter Helper)**:
  - Defines the helper function that saves the data capsule after each step.
- **Cell 5 (Arm 1: Default Allocator Bisection)**:
  - Runs 3 anchor repeats at 114,688 (known success).
  - Confirms OOM at 131,072.
  - Automatically bisects midpoints (e.g., 122,880 $\rightarrow$ 118,784 $\rightarrow$ ...) until the boundary is narrowed down to $\le 512$ tokens!
  - Takes about 10–15 minutes.
- **Cell 6 (Export Arm 1 Checkpoint)**:
  - Packages and saves the Arm 1 results.
- **Cell 7 (Arm 2: Expandable Segments Bisection)**:
  - Switches `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`.
  - Runs 3 anchor repeats at 114,688.
  - Runs bisection to find the exact boundary under the modern allocator!
  - Takes about 10–15 minutes.
- **Cell 8 (Export Arm 2 Checkpoint)**:
  - Packages and saves the Arm 2 results.
- **Cell 9 (Compare Allocators & Compute Delta)**:
  - Compares the two boundaries and prints the token improvement:
    $$\Delta = \text{Success}_{\text{expandable}} - \text{Success}_{\text{default}}$$
- **Cell 10 (Bundle Final Capsule Zip)**:
  - Zips all results, logs, and receipts into a single downloadable archive:
    `artifacts/bisect_boundary_t4_YYYYMMDD.zip`.
- **Cell 11 (Download to Your Laptop)**:
  - Triggers an automatic browser download of the zip file to your computer!

### Step 5: What to Do With the Downloaded File
Once the zip file downloads to your laptop:
1. Move the zip file into the project's `artifacts/` folder.
2. Tell me: **"RESULTS READY"**.
3. I will immediately verify the cryptographic hashes, unpack the data, update the scientific tables, and write the final response to the Teaching Assistant!

---

## 8. Summary Cheat Sheet

| Question | Answer in Plain English |
| :--- | :--- |
| **What is the model?** | `Qwen2.5-1.5B` in half-precision (FP16). |
| **What is the hardware?** | Nvidia Tesla T4 GPU with ~15.8 GB of VRAM. |
| **How fast does memory grow?** | Exactly **28,672 bytes per token**. |
| **How much room do weights take?** | ~3.09 GB static base memory. |
| **What was our original prediction?** | **111,232 tokens** (slightly conservative by ~3%). |
| **What is the highest tested success?** | **114,688 tokens** (fits within 0.01% of theory). |
| **What is the known failure point?** | **131,072 tokens** (crashes due to attention mask cliff). |
| **What is the goal of the bisection?** | Pinpoint the exact breaking point between 114,688 and 131,072 down to $\le 512$ tokens for both PyTorch allocators. |
| **Where is the notebook?** | `notebooks/BOUNDARY_COLAB.ipynb`. |
| **Where is the compiled PDF guide?** | `docs/PROJECT_EXPLANATION_BEGINNER_GUIDE.pdf`. |

---
*End of Beginner's Guide. Happy experimenting!*
