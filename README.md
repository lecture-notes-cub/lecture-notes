
# Task definition

### What the agent does?

Agent transcribes video and audio recordings into notes with optional additional context.

### For whom?

Students, who prefer text over video or want to recall the material.

### Why it has to be an agent and not a script (or a one-shot ChatGPT message)?

- Multimodal data processing which is not supported in basic ChatGPT
- Automatic text correction using context
- Agentic setup would allow for more flexibility: adding more context in form of pdfs, adding tools such as pointing at specific time ranges in the video-lecture where a specific topic is discussed, etc. This either can’t be achieved with a single script or would be hard to achieve
- We suppose that the context of each video-lecture would be too big for a one-shot prompt.
- Agent can decide to return to some part of the lecture to clarify context.

### What model(s) are you planning to use?

- In general: local models such as Qwen3-Omni-30B-A3B-Thinking
- For audio processing: Qwen3-ASR-1.7B
- For video processing: VideoLLaMA3-7B

### Which parts are you planning to solve in code? Why?

- Multimodal interactions, if we plan on using different models for different modalities
- Specific features/tools
- Some other structural bases

We want to do this in code because this behaviour would be better to be deterministic and act predictably.

# Initial architecture

- Perceive through the user input, video and audio, some additional feedback from models responsible for different modalities. This is the necessary and sufficient background for the agent to be able to do the job and not to be distracted.
- Memory approach:
  - Short-term: keeps the current topic, recent tool results, list of unresolved issues. So that the notes are consistent and each next topic follows naturally from the last.
  - Long-term: has a knowledge base of different courses and lectures the user has provided before, for every course and lecture keeps a smaller knowledge base (transcripts, notes, the curriculum, glossary, abbreviations, theorems, etc.). So that current notes are consistent with the previous ones and a recall tool can be created/used.
  - Scratchpad of the current lecture
- We assume the following structure: Act (transcribe the lecture using a tool) -> Reason (think what the notes should include) -> Act (write the notes) -> Observe (evaluate the notes, if necessary rewrite them, fix the grammar, etc).

# Failure map

1. Incorrect recognition of technical/math terms -> create predefined glossary of terms, add web-search about this theme. Figured it out through group discussion.
2. The agent may not recall the whole lecture, leaving out some vital information. Figured it out through group discussion.
3. The agent may misunderstand who’s speaking and assume that a student’s phrase/question was actually the professor’s -> add mapping of text with speaker using voice detection + context. Figured it out through group discussion.
4. Important information appears only in video or audio -> sync audio and video using timestamps. Figured it out through group discussion.
5. Bad video quality could result in the agent misunderstanding the formulas/symbols, which could harm the consistency of the notes or lead to the agent hallucinating something. Figured it out through group discussion.
6. The agent omits important details. Figured it out through group discussion.

# Initial evaluation plan

- Initial/easiest evaluations:
  - Compare metrics against labeled (topics, formulas, theorems, etc.) notes on specific video-lectures. Find the notes on the internet/ask the professors/students. Label a few ourselves, then maybe use an LLM to label more.

- Metrics:
  - **Concept Coverage (Recall)** - how many key ideas of the lecture remain in the notes. Use either simple word search or cosine similarity\perplexity in harder cases to check if the labeled ideas are in the agent’s notes.
  - **Factual Precision** - factual correctness. Using LLM as a judge.
  - **Faithfulness** - a share of claims actually grounded in the lecture. Calculate similarly to recall.
  - **Note Quality** - structure, helpfulness, clarity. Using LLM as a judge.
