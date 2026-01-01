# 🤖 Agentic RAG Study Guide
## NVIDIA Agentic AI Certification Prep - Day 2

**Focus:** Query Reformulation, Self-Correction Loops, Multi-hop Reasoning

---

## 📚 Table of Contents

1. [What is Agentic RAG?](#what-is-agentic-rag)
2. [Key Patterns](#key-patterns)
3. [Query Reformulation](#query-reformulation)
4. [Self-Correction Loops](#self-correction-loops)
5. [Multi-hop Reasoning](#multi-hop-reasoning)
6. [Retrieval Strategies](#retrieval-strategies)
7. [Grounding & Hallucination Detection](#grounding--hallucination-detection)
8. [LangGraph Implementation](#langgraph-implementation)
9. [NVIDIA Course Alignment](#nvidia-course-alignment)
10. [Hands-On Exercises](#hands-on-exercises)

---

## 🎯 What is Agentic RAG?

**Traditional RAG:**
```
Query → Retrieve → Generate → Response
```

**Agentic RAG:**
```
Query → Analyze → [Reformulate?] → Retrieve → Grade → [Retry?] → Generate → [Verify?] → Response
                      ↑                           ↑                          ↑
                      └───────────────────────────┴──────────────────────────┘
                                    Self-Correction Loops
```

### Key Differences

| Aspect | Traditional RAG | Agentic RAG |
|--------|----------------|-------------|
| Query handling | As-is | Reformulated/expanded |
| Retrieval | Single attempt | Multiple strategies |
| Document handling | All used | Relevance graded |
| Response validation | None | Grounding checked |
| Error handling | Fail | Self-correct & retry |

### NVIDIA Terminology

| Term | Definition |
|------|------------|
| **Corrective RAG (CRAG)** | Grades documents and corrects retrieval if quality is low |
| **Self-RAG** | Self-reflects on output and decides whether to retrieve more |
| **Adaptive RAG** | Chooses retrieval strategy based on query complexity |

---

## 🔑 Key Patterns

### Pattern 1: Query Analysis
```python
async def analyze_query(query: str) -> dict:
    """
    Analyze query to determine:
    - Type: simple, complex, multi_hop
    - Key entities to search
    - Optimized search query
    - Sub-queries if needed
    """
    # LLM classifies and reformulates
    ...
```

### Pattern 2: Document Grading
```python
async def grade_documents(query: str, docs: list) -> list:
    """
    Grade each document for relevance.
    Filter out irrelevant documents.
    """
    relevant = []
    for doc in docs:
        score = await llm_grade(query, doc)
        if score > threshold:
            relevant.append(doc)
    return relevant
```

### Pattern 3: Response Verification
```python
async def verify_grounding(response: str, context: str) -> bool:
    """
    Check if response is grounded in context.
    Detect potential hallucinations.
    """
    # LLM checks each claim against context
    ...
```

### Pattern 4: Self-Correction Loop
```python
async def self_correct(state: RAGState) -> RAGState:
    """
    If response is not grounded:
    1. Reformulate query
    2. Try different retrieval strategy
    3. Generate with stricter grounding
    """
    if not state.is_grounded and state.iteration < max_iterations:
        state.query = reformulate(state.query, state.issues)
        state.strategy = next_strategy(state.strategy)
        return await rag_pipeline(state)
    return state
```

---

## 📝 Query Reformulation

### Why Reformulate?

Users often ask questions that don't match how information is stored:

| User Query | Stored As |
|------------|-----------|
| "Can I paint my house blue?" | "Exterior paint colors must be approved..." |
| "What's the fence height limit?" | "Fences shall not exceed 6 feet..." |
| "How do I complain about my neighbor?" | "Violation reporting process..." |

### Reformulation Techniques

#### 1. Search Query Optimization
```python
# Original: "Can I build a shed in my backyard?"
# Optimized: "shed outbuilding accessory structure backyard requirements approval"
```

#### 2. Query Expansion
```python
# Original: "fence rules"
# Expanded: [
#     "fence rules",
#     "fencing requirements regulations",
#     "fence height material guidelines",
#     "property boundary fence specifications"
# ]
```

#### 3. Entity Extraction
```python
# Query: "What color can I paint my fence if it's on the property line?"
# Entities: ["fence", "paint color", "property line"]
# Search each entity separately, combine results
```

### Implementation

```python
QUERY_REFORMULATION_PROMPT = """
Rewrite this query for better semantic search results.

Original: {query}

Instructions:
- Include synonyms and related terms
- Remove unnecessary words
- Focus on key concepts
- Make it match how documents are typically written

Optimized query:
"""
```

---

## 🔄 Self-Correction Loops

### Loop Structure

```
┌─────────────┐
│   START     │
└──────┬──────┘
       ▼
┌─────────────┐
│  Retrieve   │◄─────────────┐
└──────┬──────┘              │
       ▼                     │
┌─────────────┐              │
│   Grade     │              │
│  Documents  │              │
└──────┬──────┘              │
       ▼                     │
   ┌───┴───┐                 │
   │Relevant│                │
   │ docs?  │                │
   └───┬───┘                 │
    No │ Yes                 │
       │  ▼                  │
       │ ┌─────────────┐     │
       │ │  Generate   │     │
       │ └──────┬──────┘     │
       │        ▼            │
       │    ┌───┴───┐        │
       │    │Grounded│        │
       │    │   ?    │        │
       │    └───┬───┘        │
       │     No │ Yes        │
       │        │  ▼         │
       │        │ ┌──────┐   │
       │        │ │ END  │   │
       │        │ └──────┘   │
       ▼        ▼            │
┌─────────────────┐          │
│    Reformulate  │──────────┘
│      Query      │
└─────────────────┘
```

### Correction Strategies

| Iteration | Strategy Change |
|-----------|-----------------|
| 1 | Semantic search with reformulated query |
| 2 | Expanded query (synonyms + variations) |
| 3 | Hybrid search (semantic + keyword) |
| 4+ | Acknowledge limitation, partial answer |

### Implementation

```python
async def self_correction_loop(
    query: str,
    max_iterations: int = 3,
) -> RAGResult:
    """Execute RAG with self-correction."""
    
    state = RAGState(query=query)
    
    for i in range(max_iterations):
        # Retrieve
        state.documents = await retrieve(state.search_query)
        
        # Grade documents
        state.relevant_docs = await grade(state.documents)
        
        if not state.relevant_docs:
            # No relevant docs - reformulate
            state.search_query = await reformulate(
                query, 
                issue="No relevant documents found"
            )
            state.strategy = next_strategy(state.strategy)
            continue
        
        # Generate
        state.response = await generate(query, state.relevant_docs)
        
        # Check grounding
        grounded, confidence, issues = await check_grounding(
            state.response, 
            state.relevant_docs
        )
        
        if grounded and confidence > 0.8:
            return state  # Success!
        
        # Not grounded - correct and retry
        state.search_query = await reformulate(query, issue=issues)
        state.previous_response = state.response
    
    return state  # Return best effort
```

---

## 🔗 Multi-hop Reasoning

### When to Use

Multi-hop is needed when answering requires:
1. Sequential information lookup
2. Combining facts from multiple sources
3. Reasoning chains where step N depends on step N-1

### Example

**Query:** "Can I build the same type of fence my neighbor built last year?"

**Hop 1:** What type of fence did the neighbor build?
- Requires lookup in: Approval records, violation reports

**Hop 2:** What are the current fence requirements?
- Requires lookup in: CC&Rs, ARC guidelines

**Hop 3:** Are there any changes since last year?
- Requires lookup in: Recent amendments, board decisions

**Final:** Combine all information for answer

### Implementation

```python
async def multi_hop_rag(query: str) -> RAGResult:
    """Execute multi-hop retrieval."""
    
    # Step 1: Decompose into sub-queries
    analysis = await analyze_query(query)
    sub_queries = analysis["sub_queries"]
    
    # Step 2: Execute each sub-query
    all_context = []
    for sub_q in sub_queries:
        result = await single_hop_rag(sub_q)
        all_context.append({
            "query": sub_q,
            "answer": result.response,
            "sources": result.documents,
        })
    
    # Step 3: Synthesize final answer
    final_response = await synthesize(query, all_context)
    
    return RAGResult(
        query=query,
        response=final_response,
        sub_queries=sub_queries,
    )
```

---

## 🎯 Retrieval Strategies

### Strategy Comparison

| Strategy | Best For | Tradeoffs |
|----------|----------|-----------|
| **Semantic** | Conceptual matches | May miss exact terms |
| **Keyword** | Exact term matches | Misses synonyms |
| **Hybrid** | Balanced | More complex |
| **Expanded** | Comprehensive | Slower, may retrieve noise |

### Semantic Search
```python
# Dense vector similarity
query_embedding = embed(query)
results = vector_db.search(query_embedding, top_k=5)
```

### Keyword Search
```python
# BM25 or similar
results = full_text_search(query, top_k=5)
```

### Hybrid Search
```python
# Combine both with score fusion
semantic_results = vector_search(query)
keyword_results = text_search(query)
results = reciprocal_rank_fusion(semantic_results, keyword_results)
```

### Query Expansion
```python
# Search with multiple variations
variations = generate_variations(query)
all_results = []
for var in variations:
    results = vector_search(var)
    all_results.extend(results)
# Deduplicate and re-rank
```

---

## ✅ Grounding & Hallucination Detection

### What is Grounding?

A response is **grounded** if every claim is supported by the retrieved context.

### Hallucination Types

| Type | Example |
|------|---------|
| **Fabrication** | Inventing facts not in context |
| **Exaggeration** | "Always" when context says "usually" |
| **Misattribution** | Attributing to wrong source |
| **Extrapolation** | Drawing conclusions beyond evidence |

### Detection Approach

```python
GROUNDING_CHECK_PROMPT = """
For each claim in the response, determine if it's supported by the context.

Context:
{context}

Response:
{response}

For each claim:
1. Is it directly stated in the context? (SUPPORTED)
2. Is it a reasonable inference? (INFERRED)  
3. Is it not in the context? (UNSUPPORTED)

List any UNSUPPORTED claims.
"""
```

### Self-Correction for Hallucinations

```python
async def correct_hallucination(
    query: str,
    response: str,
    issues: list[str],
    context: str,
) -> str:
    """Generate corrected response."""
    
    prompt = f"""
    The previous response had unsupported claims:
    {issues}
    
    Generate a new response that:
    1. Only uses information from the context
    2. Explicitly states when information is unavailable
    3. Avoids the identified issues
    
    Context: {context}
    Query: {query}
    """
    
    return await llm.generate(prompt)
```

---

## 🔄 LangGraph Implementation

### State Definition

```python
class RAGState(TypedDict):
    """State for agentic RAG graph."""
    
    # Input
    query: str
    
    # Query processing
    query_type: str
    reformulated_query: str
    sub_queries: list[str]
    
    # Retrieval
    documents: list[Document]
    relevant_documents: list[Document]
    retrieval_strategy: str
    
    # Generation
    response: str
    
    # Verification
    is_grounded: bool
    confidence: float
    issues: str
    
    # Control
    iteration: int
    max_iterations: int
```

### Graph Structure

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(RAGState)

# Add nodes
workflow.add_node("analyze", analyze_query)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generate_response)
workflow.add_node("verify", verify_grounding)
workflow.add_node("rewrite", rewrite_query)

# Set entry point
workflow.set_entry_point("analyze")

# Add edges
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "grade")

# Conditional edges
workflow.add_conditional_edges(
    "grade",
    decide_after_grading,
    {"generate": "generate", "rewrite": "rewrite", "end": END}
)

workflow.add_edge("generate", "verify")

workflow.add_conditional_edges(
    "verify",
    decide_after_verification,
    {"end": END, "retry": "rewrite"}
)

workflow.add_edge("rewrite", "retrieve")

# Compile
graph = workflow.compile()
```

### Conditional Functions

```python
def decide_after_grading(state: RAGState) -> str:
    """Decide next step after document grading."""
    if state["relevant_documents"]:
        return "generate"
    elif state["iteration"] < state["max_iterations"]:
        return "rewrite"
    else:
        return "end"

def decide_after_verification(state: RAGState) -> str:
    """Decide next step after grounding check."""
    if state["is_grounded"] and state["confidence"] >= 0.8:
        return "end"
    elif state["iteration"] < state["max_iterations"]:
        return "retry"
    else:
        return "end"
```

---

## 📖 NVIDIA Course Alignment

### "Building RAG Agents With LLMs" Course

| Topic | Implementation |
|-------|----------------|
| Agent retrieval capability | `AgenticRAG.retrieve()` |
| Tool use for documents | Custom Milvus retriever |
| Planning approaches | Query analysis & decomposition |
| Scaling considerations | Iteration limits, early termination |

### Key Exam Concepts

1. **Query Routing** - Classify query complexity
2. **Document Grading** - LLM-based relevance scoring
3. **Response Verification** - Grounding checks
4. **Iterative Refinement** - Self-correction loops
5. **State Management** - LangGraph for control flow

---

## 🧪 Hands-On Exercises

### Exercise 1: Query Reformulation

```python
# Test query reformulation
queries = [
    "Can I have chickens?",
    "What about my fence?",
    "Is 7 feet too tall?",
    "neighbor's tree",
]

for q in queries:
    reformulated = await rag.reformulate_query(q)
    print(f"Original: {q}")
    print(f"Reformulated: {reformulated}")
    print("---")
```

### Exercise 2: Document Grading

```python
# Test document grading
query = "What are the fence height limits?"

docs = await rag.retrieve(query)
graded = await rag.check_relevance(query, docs)

print(f"Retrieved: {len(docs)}")
print(f"Relevant: {len(graded)}")
for doc in graded:
    print(f"  - {doc.source}: {doc.score:.2f}")
```

### Exercise 3: Self-Correction

```python
# Test self-correction loop
query = "Can I install solar panels on my detached garage?"

result = await rag.query(query)

print(f"Iterations: {result.iterations}")
print(f"Strategy: {result.strategy_used}")
print(f"Grounded: {result.is_grounded}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Response: {result.response}")
```

### Exercise 4: Multi-hop Reasoning

```python
# Test multi-hop query
query = "What modifications require ARC approval and how long does approval take?"

result = await rag.multi_hop_query(query)

print(f"Sub-queries: {result.sub_queries}")
print(f"Documents used: {len(result.retrieved_chunks)}")
print(f"Response: {result.response}")
```

---

## ✅ Study Checklist

### Query Reformulation
- [ ] Understand why reformulation improves retrieval
- [ ] Know techniques: optimization, expansion, entity extraction
- [ ] Implement LLM-based query reformulation

### Self-Correction
- [ ] Understand the correction loop structure
- [ ] Know when to retry vs. terminate
- [ ] Implement iteration limits and strategy rotation

### Multi-hop Reasoning
- [ ] Identify when multi-hop is needed
- [ ] Decompose queries into sub-queries
- [ ] Aggregate sub-query results

### Grounding
- [ ] Define grounding vs. hallucination
- [ ] Implement grounding checks
- [ ] Correct ungrounded responses

### LangGraph
- [ ] Define state for RAG pipeline
- [ ] Implement nodes and conditional edges
- [ ] Compile and execute graph

---

## 🎯 Exam Tips

1. **Know the terminology** - CRAG, Self-RAG, Adaptive RAG
2. **Understand control flow** - When to retry, when to stop
3. **State management** - Track iteration, strategy, confidence
4. **Quality over quantity** - Better to have fewer grounded docs than many irrelevant ones
5. **Graceful degradation** - Always have a fallback response

---

**Next:** Day 3 - RAGAS Evaluation 📊

*Last updated: December 29, 2024*