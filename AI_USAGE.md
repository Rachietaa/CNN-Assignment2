# AI Usage

## 1. AI coding tool used

**Claude Code** (Anthropic), used inside VS Code as a pair programmer for the whole project.

As the assignment puts it, the AI was my pair programmer, not my research scientist. It wrote code I
asked for, debugged errors, explained SSL methods and **suggested** options. I decided which SSL method to
investigate, which experiments to run and how to make the comparisons fair, interpreted the results, and
made every Git commit myself, step by step. Each decision and the reason behind it is recorded in
[DECISIONS.md](DECISIONS.md).

---

## 2. Representative examples of how AI assisted me

*(Filled in as the project progresses.)*

---

## 3. An incorrect, ineffective or questionable AI suggestion

*(Filled in as the project progresses.)*

---

## 4. How I verified or modified the AI-generated solutions

| Issue | How it was verified | Change made |
|---|---|---|
| | | |

---

## 5. Decisions I made rather than accepting the AI's recommendation

### Writing fresh code instead of copying the Assignment 1 pipeline

The AI's first plan was to copy the Assignment 1 code (data loading, training loop, configs) into this
repository and extend it. I chose to start a fresh codebase and reuse only the **data** and the **same
train/validation split** from Assignment 1. SSL needs different components anyway (two augmented views per
image, a contrastive loss, a separate pretraining loop), and a fresh repository contains only code that is
actually used in this assignment.
