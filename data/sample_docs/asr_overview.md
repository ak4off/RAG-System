# Automatic Speech Recognition (ASR) Overview

## What is ASR?

Automatic Speech Recognition (ASR) is the task of converting spoken audio into text. It is used in applications such as voice assistants, transcription systems, and call center analytics.

---

## Core Challenges

ASR systems must handle:

* Variability in accents and pronunciation
* Background noise
* Speaker differences
* Long-form audio with unclear boundaries

---

## Connectionist Temporal Classification (CTC)

CTC (Connectionist Temporal Classification) is a training objective used in sequence modeling when alignment between input and output is unknown.

Key idea:

* Allows the model to predict sequences without explicit frame-level labels
* Introduces a special blank token
* Uses dynamic programming to compute probabilities over all alignments

---

## Encoder Architectures

### Transformer Encoder

Uses self-attention to model global context across the input sequence.

### Conformer

Combines convolution (local context) and attention (global context).

### E-Branchformer

A variant that processes input through parallel branches:

* Convolution branch for local patterns
* Attention branch for long-range dependencies

---

## Decoding Strategies

### Greedy Decoding

Selects the most probable token at each step. Fast but suboptimal.

### Beam Search

Maintains multiple hypotheses and selects the best sequence globally.

---

## Evaluation Metrics

### Word Error Rate (WER)

WER measures transcription accuracy:

WER = (S + D + I) / N

Where:

* S = substitutions
* D = deletions
* I = insertions
* N = number of words in reference

---

### Character Error Rate (CER)

CER is similar to WER but operates at the character level.

Used for:

* languages without clear word boundaries
* morphologically rich languages (e.g., Tamil)

---

## Real-Time Factor (RTF)

RTF = processing time / audio duration

* RTF < 1 → faster than real-time
* RTF > 1 → slower than real-time

---

## Production ASR Pipeline

A typical pipeline includes:

1. Voice Activity Detection (VAD)

   * Removes silence segments

2. Chunking

   * Splits long audio into overlapping windows

3. Acoustic Model

   * Converts audio to token probabilities

4. Language Model (optional)

   * Improves fluency

5. Post-processing

   * Punctuation, casing, formatting

---

## Sliding Window Inference

Long audio is processed using overlapping chunks:

* Example: 30s window with 5s overlap
* Overlap ensures continuity
* Outputs are stitched together

---

## Key Trade-offs

* Accuracy vs latency
* Model size vs deployment cost
* Greedy vs beam search decoding

---

## Summary

ASR systems combine signal processing, deep learning, and language modeling to produce accurate transcriptions. Modern systems rely heavily on transformer-based architectures and efficient decoding strategies.
