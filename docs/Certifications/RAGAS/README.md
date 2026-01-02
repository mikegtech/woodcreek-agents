# 📊 RAGAS Evaluation Study Guide

## NVIDIA Agentic AI Certification - Day 3

---

## 🎯 Why Evaluation Matters

> "The difference between people that really know how to build agentic workflows compared to people that are less effective at it is the ability to drive a disciplined development process, specifically one focused on evals and error analysis."

Evaluation is the foundation of production-quality AI systems:
- **Without evals**: You're flying blind, shipping hope
- **With evals**: You have data-driven improvement, measurable progress

---

## 📚 RAGAS Overview

**RAGAS** = **R**etrieval **A**ugmented **G**eneration **A**ssessment

Key insight: RAGAS evaluates RAG systems **without requiring ground truth for every question**. This is critical because creating ground truth is expensive.

### The Four Pillars

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Quality Metrics                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   RETRIEVAL QUALITY          GENERATION QUALITY              │
│   ┌─────────────────┐       ┌─────────────────┐             │
│   │ Context         │       │ Faithfulness    │             │
│   │ Precision       │       │                 │             │
│   │                 │       │ Is answer       │             │
│   │ Are retrieved   │       │ supported by    │             │
│   │ docs relevant?  │       │ context?        │             │
│   └─────────────────┘       └─────────────────┘             │
│                                                              │
│   ┌─────────────────┐       ┌─────────────────┐             │
│   │ Context         │       │ Answer          │             │
│   │ Recall          │       │ Relevancy       │             │
│   │                 │       │                 │             │
│   │ Did we get all  │       │ Does answer     │             │
│   │ needed docs?    │       │ address the     │             │
│   │                 │       │ question?       │             │
│   └─────────────────┘       └─────────────────┘             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📏 The Metrics Deep Dive

### 1. Faithfulness (Most Critical)

**What it measures**: Is every claim in the answer supported by the retrieved context?

**Formula**:
```
Faithfulness = Supported Claims / Total Claims
```

**Process**:
1. Extract factual claims from the answer
2. For each claim, check if context supports it
3. Calculate ratio of supported claims

**Example**:
```
Question: "What is the fence height limit?"
Context: "Fences shall not exceed 6 feet in height."
Answer: "The maximum fence height is 6 feet, and violations result in $500 fines."

Claims:
1. "Maximum fence height is 6 feet" → SUPPORTED ✅
2. "Violations result in $500 fines" → NOT SUPPORTED ❌

Faithfulness = 1/2 = 0.5
```

**Why it matters**: Low faithfulness = HALLUCINATION

**How to improve**:
- Strengthen grounding prompts
- Add citation requirements
- Increase relevance thresholds
- Use smaller, focused chunks

---

### 2. Answer Relevancy

**What it measures**: Does the answer address what the user asked?

**Formula**:
```
Relevancy = avg(similarity(original_question, generated_questions_from_answer))
```

**Process**:
1. Generate N questions the answer would be appropriate for
2. Compute semantic similarity between original question and generated questions
3. Average the similarities

**Example**:
```
Question: "What is the fence height limit?"
Answer: "ARC approval is required for all exterior modifications."

Generated questions from answer:
- "What requires ARC approval?"
- "What are the rules for exterior modifications?"

Similarity to original question: LOW
Relevancy score: 0.3 (answer doesn't address the question!)
```

**Why it matters**: High faithfulness + low relevancy = correct but useless

**How to improve**:
- Better query understanding
- Add query analysis step
- Explicit instructions to address the question

---

### 3. Context Precision

**What it measures**: Are the retrieved documents actually relevant to the question?

**Formula**:
```
Precision = Σ(relevance × position_weight) / Σ(position_weight)
```

Higher-ranked documents have higher weight (position 1 > position 5)

**Process**:
1. For each retrieved document, check if it's relevant
2. Weight by position (earlier = more important)
3. Calculate weighted precision

**Example**:
```
Question: "What is the fence height limit?"

Retrieved documents:
1. "Fences shall not exceed 6 feet" → RELEVANT ✅ (weight: 1.0)
2. "Paint colors must be approved" → NOT RELEVANT ❌ (weight: 0.5)
3. "Board meeting minutes from March" → NOT RELEVANT ❌ (weight: 0.33)

Precision = (1.0 × 1 + 0.5 × 0 + 0.33 × 0) / (1.0 + 0.5 + 0.33) = 0.55
```

**Why it matters**: Low precision = noisy retrieval confusing the generator

**How to improve**:
- Better query reformulation
- Higher relevance thresholds
- Metadata filtering
- Better embeddings

---

### 4. Context Recall

**What it measures**: Did we retrieve all the information needed to answer correctly?

**Formula**:
```
Recall = Facts Found in Context / Total Facts in Ground Truth
```

**⚠️ Requires ground truth answer**

**Process**:
1. Extract key facts from ground truth
2. Check if each fact is present in retrieved context
3. Calculate ratio of found facts

**Example**:
```
Question: "What is the ARC approval process?"
Ground Truth: "Submit form ARF-1, pay $50 fee, wait 30 days for decision."

Key facts:
1. "Submit form ARF-1" → FOUND ✅
2. "Pay $50 fee" → NOT FOUND ❌
3. "Wait 30 days" → FOUND ✅

Recall = 2/3 = 0.67
```

**Why it matters**: Low recall = missing critical information

**How to improve**:
- Increase top_k
- Query expansion
- Hybrid search (semantic + keyword)
- Better chunking

---

## 🔬 Metric Interactions

```
                    ┌──────────────────┐
                    │ Overall Quality  │
                    └────────┬─────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │ Retrieval   │   │ Generation  │   │ End-to-End  │
    │ Quality     │   │ Quality     │   │ Experience  │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
      ┌────┴────┐       ┌────┴────┐            │
      │         │       │         │            │
      ▼         ▼       ▼         ▼            ▼
   Precision  Recall  Faith.  Relevancy    All Four!
```

### Diagnostic Patterns

| Precision | Recall | Faithfulness | Relevancy | Diagnosis |
|-----------|--------|--------------|-----------|-----------|
| Low | High | * | * | Noisy retrieval |
| High | Low | * | * | Missing documents |
| High | High | Low | * | Hallucination problem |
| High | High | High | Low | Answer generation problem |
| High | High | High | High | ✅ System working well! |

---

## 🛠️ Implementation Patterns

### Pattern 1: Claim Extraction for Faithfulness

```python
async def extract_claims(answer: str) -> list[str]:
    """Extract verifiable claims from an answer."""
    prompt = """
    Extract all factual claims from this answer.
    A claim is a statement that can be verified as true or false.
    
    Answer: {answer}
    
    Claims (one per line):
    """
    # Use LLM to extract claims
    claims = await llm.generate(prompt.format(answer=answer))
    return parse_claims(claims)
```

### Pattern 2: Claim Verification

```python
async def verify_claim(claim: str, context: str) -> bool:
    """Check if claim is supported by context."""
    prompt = """
    Is this claim supported by the context?
    
    Context: {context}
    Claim: {claim}
    
    Answer SUPPORTED or NOT_SUPPORTED:
    """
    result = await llm.generate(prompt.format(claim=claim, context=context))
    return "SUPPORTED" in result.upper()
```

### Pattern 3: Embedding-based Relevancy

```python
def compute_relevancy(question: str, answer: str) -> float:
    """Compute answer relevancy using embeddings."""
    # Generate questions the answer would address
    gen_questions = generate_questions_from_answer(answer)
    
    # Embed original and generated questions
    orig_emb = embed(question)
    gen_embs = [embed(q) for q in gen_questions]
    
    # Average cosine similarity
    similarities = [cosine_sim(orig_emb, emb) for emb in gen_embs]
    return sum(similarities) / len(similarities)
```

---

## 📊 Evaluation Workflow

### Step 1: Create Evaluation Dataset

```python
# Define test questions (with optional ground truth)
questions = [
    {
        "question": "What is the fence height limit?",
        "ground_truth": "6 feet",  # Optional but enables recall
    },
    {
        "question": "What is ARC approval?",
        # No ground truth - still evaluates faithfulness, precision, relevancy
    },
]

# Generate dataset by running RAG
dataset = await generator.generate_dataset(questions)
```

### Step 2: Run Evaluation

```python
evaluator = RAGASEvaluator()
report = await evaluator.evaluate_dataset(dataset)
```

### Step 3: Analyze Results

```python
evaluator.print_report(report)

# Output:
# RAGAS EVALUATION REPORT
# ========================
# Faithfulness:       0.85 ✅
# Answer Relevancy:   0.78 ✅
# Context Precision:  0.62 ⚠️  <-- Problem!
# Context Recall:     0.71 ✅
# ------------------------
# OVERALL SCORE:      0.74 ✅
```

### Step 4: Error Analysis

```python
# Find problem samples
for i in report.low_precision_samples:
    sample = report.sample_evaluations[i].sample
    print(f"Low precision query: {sample.question}")
    print(f"Retrieved contexts: {sample.contexts}")
    # Investigate why irrelevant docs were retrieved
```

### Step 5: Iterate

Based on analysis:
- Low faithfulness → Improve grounding
- Low precision → Improve retrieval
- Low relevancy → Improve generation
- Low recall → Expand search

---

## 🎯 Exam Key Points

### Must Know:
1. **Four metrics**: Faithfulness, Answer Relevancy, Context Precision, Context Recall
2. **Which require ground truth**: Only Context Recall
3. **What each measures**: Retrieval vs Generation quality
4. **How to interpret scores**: >0.7 typically passing
5. **How to improve each metric**: Specific interventions

### Common Exam Questions:

**Q: Your RAG system has high faithfulness but low answer relevancy. What's wrong?**
A: The answers are factually correct but not addressing the questions. Improve query understanding and answer generation prompts.

**Q: You observe low context precision. What should you investigate?**
A: The retrieval is returning irrelevant documents. Check query reformulation, relevance thresholds, and embedding quality.

**Q: Why is faithfulness the most critical metric?**
A: Because low faithfulness means hallucination - the system is making up information not in the source documents.

**Q: Can you evaluate context recall without ground truth?**
A: No. Context recall requires ground truth to know what information should have been retrieved.

---

## 📝 Quick Reference

| Metric | Measures | Requires GT | Improves With |
|--------|----------|-------------|---------------|
| Faithfulness | Grounding | No | Better prompts, stricter filtering |
| Answer Relevancy | Addressing question | No | Query analysis, generation prompts |
| Context Precision | Retrieval quality | No | Better embeddings, thresholds |
| Context Recall | Retrieval coverage | **Yes** | Higher top_k, query expansion |

---

## 🔗 Resources

- [RAGAS Paper](https://arxiv.org/abs/2309.15217)
- [RAGAS Documentation](https://docs.ragas.io/)
- [LangChain Evaluation](https://python.langchain.com/docs/guides/evaluation)

---

## ✅ Day 3 Checklist

- [ ] Understand all four RAGAS metrics
- [ ] Know which metrics require ground truth
- [ ] Understand the evaluation pipeline
- [ ] Know how to interpret scores
- [ ] Know improvement strategies for each metric
- [ ] Understand metric interactions
- [ ] Practice with the API endpoints

---

**Next: Day 4-6 - Course Review and Practice Exams**