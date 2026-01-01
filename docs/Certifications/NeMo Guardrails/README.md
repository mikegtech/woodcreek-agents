# 🛡️ NeMo Guardrails Study Guide
## NVIDIA Agentic AI Certification Prep

**Exam Date:** January 4, 2025  
**Version:** NeMo Guardrails 0.19.x  
**Colang Version:** 2.0 (stable)

---

## 📚 Table of Contents

1. [Core Concepts](#core-concepts)
2. [Colang 2.0 Language](#colang-20-language)
3. [Types of Guardrails](#types-of-guardrails)
4. [Integration Patterns](#integration-patterns)
5. [NVIDIA NIM Integration](#nvidia-nim-integration)
6. [Agentic AI Safety](#agentic-ai-safety)
7. [Official Resources](#official-resources)
8. [Hands-On Labs](#hands-on-labs)
9. [Study Checklist](#study-checklist)

---

## 🎯 Core Concepts

### What is NeMo Guardrails?

NeMo Guardrails is an **open-source toolkit** for adding programmable guardrails to LLM-based conversational systems. It controls LLM output in specific ways:

- Not talking about certain topics (politics, competitors)
- Responding in a particular way to specific requests
- Following predefined dialog paths
- Using a particular language style
- Extracting structured data
- Preventing jailbreaks and prompt injections

### Key Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input                                │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 INPUT RAILS                                  │
│  • Jailbreak detection                                       │
│  • Content safety                                            │
│  • Topic control                                             │
│  • PII masking                                               │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 DIALOG RAILS                                 │
│  • Conversation flow control                                 │
│  • Intent classification                                     │
│  • Response selection                                        │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              RETRIEVAL RAILS (RAG)                           │
│  • Relevance checking                                        │
│  • Chunk filtering                                           │
│  • Context validation                                        │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM                                       │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                OUTPUT RAILS                                  │
│  • Hallucination detection                                   │
│  • Content moderation                                        │
│  • Fact checking                                             │
│  • Response validation                                       │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Bot Response                                │
└─────────────────────────────────────────────────────────────┘
```

### Five Types of Guardrails

| Type | When Applied | Purpose |
|------|--------------|---------|
| **Input Rails** | Before processing | Reject/alter user input (jailbreak, PII masking) |
| **Dialog Rails** | During prompting | Control conversation flow, intent routing |
| **Retrieval Rails** | During RAG | Filter/validate retrieved chunks |
| **Execution Rails** | During actions | Validate tool inputs/outputs |
| **Output Rails** | Before response | Check/filter bot responses |

---

## 🗣️ Colang 2.0 Language

### Key Differences from Colang 1.0

| Feature | Colang 1.0 | Colang 2.0 |
|---------|------------|------------|
| Entry point | Implicit | `flow main` (required) |
| Flow definition | `define flow` | `flow` |
| Flow activation | Automatic | Explicit via `activate` |
| Actions | `execute` | `await` / `start` |
| Stop flow | `stop` | `abort` |
| Variables | Global by default | Local by default |
| String interpolation | `$name` | `{$name}` |
| Imports | None | Python-like imports |

### Basic Syntax

```colang
# Imports
import core
import llm
from guardrails import *

# Main flow (entry point)
flow main
    """Entry point - activates all flows."""
    activate greeting
    activate check_jailbreak
    activate llm_continuation

# User intent definition
@meta(user_intent=True)
flow user expressed greeting
    user said "hi"
    or user said "hello"
    or user said "hey"

# Bot response definition
flow bot express greeting
    bot say "Hello! How can I help you today?"

# Conversation flow
flow greeting
    user expressed greeting
    bot express greeting

# Guardrail flow with action
flow check_jailbreak
    user said $text
    $is_safe = await check_jailbreak_action(input=$text)
    if not $is_safe
        bot say "I can't help with that request."
        abort

# LLM fallback
flow llm_continuation
    user said something unexpected
    $response = await GenerateResponseAction
    bot say $response
```

### Flow Activation Patterns

```colang
# Activate single flow
activate greeting

# Activate with parameters
activate check_input with threshold=0.8

# Deactivate a flow
deactivate greeting

# Conditional activation
if $user_authenticated
    activate premium_features
```

### Actions (Async Operations)

```colang
# Blocking action (waits for result)
$result = await SomeAction(param="value")

# Non-blocking action (fire and forget)
start LoggingAction(event="user_query")

# Parallel actions
start ActionA
start ActionB
await ActionA
await ActionB
```

---

## 🚧 Types of Guardrails

### 1. Input Rails

**Purpose:** Validate and filter user input before processing

```colang
flow self check input
    """Block harmful or manipulative inputs."""
    user said $text
    $allowed = await self_check_input(user_input=$text)
    if not $allowed
        bot say "I can't process that request."
        abort
```

**Common Input Rails:**
- Jailbreak detection
- Prompt injection prevention
- Toxicity filtering
- PII detection/masking
- Topic classification

### 2. Dialog Rails

**Purpose:** Control conversation flow and intent routing

```colang
flow route_to_specialist
    """Route based on user intent."""
    user asked about $topic
    if $topic == "billing"
        activate billing_agent
    elif $topic == "technical"
        activate tech_support_agent
    else
        activate general_agent
```

### 3. Retrieval Rails (RAG)

**Purpose:** Validate retrieved context before using in prompts

```colang
flow check_retrieval_relevance
    """Ensure retrieved chunks are relevant."""
    $chunks = await RetrieveAction(query=$user_query)
    $relevant_chunks = await filter_relevant_chunks(
        chunks=$chunks,
        query=$user_query,
        threshold=0.7
    )
    if len($relevant_chunks) == 0
        bot say "I don't have relevant information on that topic."
        abort
```

### 4. Output Rails

**Purpose:** Validate bot responses before sending

```colang
flow self check output
    """Validate bot responses."""
    bot said $text
    $is_safe = await check_output_safety(response=$text)
    if not $is_safe
        bot say "Let me rephrase that response."
        abort
    
    $is_factual = await check_hallucination(
        response=$text,
        context=$retrieved_context
    )
    if not $is_factual
        bot say "I'm not certain about that. Let me verify."
        abort
```

### 5. Execution Rails

**Purpose:** Validate tool/action inputs and outputs

```colang
flow safe_tool_execution
    """Validate before executing tools."""
    $tool_input = ...
    $is_safe = await validate_tool_input(input=$tool_input)
    if not $is_safe
        bot say "I can't execute that action."
        abort
    
    $result = await ExecuteToolAction(input=$tool_input)
    
    $output_safe = await validate_tool_output(output=$result)
    if not $output_safe
        bot say "The tool returned unexpected results."
        abort
```

---

## 🔗 Integration Patterns

### LangChain Integration

```python
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

# Load config
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

# Wrap LangChain chain with guardrails
guarded_chain = RunnableRails(config) | your_chain

# Use in LangChain pipeline
result = await guarded_chain.ainvoke({"input": "user message"})
```

### LangGraph Integration (Multi-Agent)

```python
from langgraph.graph import StateGraph
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

# Create guardrailed node
def guarded_agent_node(state):
    rails = RunnableRails(config)
    response = rails.invoke(state["messages"])
    return {"messages": [response]}

# Add to LangGraph
graph = StateGraph(AgentState)
graph.add_node("guarded_agent", guarded_agent_node)
```

### Streaming Support

```python
from nemoguardrails import LLMRails

rails = LLMRails(config)

# Streaming with guardrails
async for chunk in rails.stream_async(
    messages=[{"role": "user", "content": "Hello"}]
):
    print(chunk, end="", flush=True)
```

---

## 🚀 NVIDIA NIM Integration

### NemoGuard Models (NIMs)

| Model | Purpose | Use Case |
|-------|---------|----------|
| **NemoGuard JailbreakDetect** | Detect jailbreak attempts | Input rail |
| **NemoGuard ContentSafety** | Content moderation | Input/Output rail |
| **NemoGuard TopicControl** | Topic classification | Dialog rail |

### Configuration with NIMs

```yaml
# config.yml
models:
  - type: main
    engine: nvidia_ai_endpoints
    model: meta/llama-3.1-8b-instruct

  - type: jailbreak_detection
    engine: nvidia_ai_endpoints
    model: nvidia/nemoguard-jailbreak-detect

  - type: content_safety
    engine: nvidia_ai_endpoints
    model: nvidia/llama-3.1-nemoguard-8b-content-safety

rails:
  input:
    flows:
      - jailbreak detection
      - content safety check
  output:
    flows:
      - content safety check
```

---

## 🤖 Agentic AI Safety

### Key Concepts for Exam

1. **Tool Use Safety**
   - Validate tool inputs before execution
   - Validate tool outputs before using in responses
   - Limit tool capabilities (sandboxing)

2. **Multi-Agent Coordination**
   - Guardrails at agent boundaries
   - Shared safety policies across agents
   - Hierarchical guardrails (supervisor → agents)

3. **Reasoning Trace Safety**
   - BotThinking events for monitoring LLM reasoning
   - Guardrails on chain-of-thought outputs

4. **Data Flywheel Pattern**
   - Continuous improvement of guardrails
   - Feedback loops for false positives/negatives
   - Model distillation for faster guardrails

### Agentic RAG with Guardrails

```colang
flow agentic_rag
    """RAG with self-correction and guardrails."""
    user said $query
    
    # Step 1: Query reformulation
    $refined_query = await reformulate_query(query=$query)
    
    # Step 2: Retrieve with relevance check
    $chunks = await retrieve_with_guardrails(query=$refined_query)
    
    # Step 3: Generate with hallucination check
    $response = await generate_with_fact_check(
        query=$query,
        context=$chunks
    )
    
    # Step 4: Self-correction if needed
    if not $response.is_grounded
        $response = await self_correct(
            query=$query,
            previous_response=$response
        )
    
    bot say $response.text
```

---

## 📖 Official Resources

### Documentation
- [NeMo Guardrails Documentation](https://docs.nvidia.com/nemo/guardrails/latest/)
- [Colang 2.0 Guide](https://docs.nvidia.com/nemo/guardrails/colang_2/overview.html)
- [What's Changed (1.0 → 2.0)](https://docs.nvidia.com/nemo/guardrails/colang_2/whats-changed.html)

### GitHub
- [NeMo Guardrails Repository](https://github.com/NVIDIA-NeMo/Guardrails)
- [Example Configurations](https://github.com/NVIDIA-NeMo/Guardrails/tree/main/examples)
- [Example Bots](https://github.com/NVIDIA-NeMo/Guardrails/tree/main/nemoguardrails/examples)

### Key Blog Posts (Priority Reading)

| Priority | Title | Topics |
|----------|-------|--------|
| ⭐⭐⭐ | [Safeguard Agentic AI Systems with NVIDIA Safety Recipe](https://developer.nvidia.com/blog/safeguard-agentic-ai-systems-with-the-nvidia-safety-recipe/) | Agentic safety, tool use |
| ⭐⭐⭐ | [How to Safeguard AI Agents for Customer Service](https://developer.nvidia.com/blog/how-to-safeguard-ai-agents-for-customer-service/) | End-to-end tutorial |
| ⭐⭐⭐ | [Content Moderation with NeMo Guardrails](https://developer.nvidia.com/blog/content-moderation-and-safety-checks-with-nvidia-nemo-guardrails/) | RAG safety |
| ⭐⭐ | [Secure Generative AI with NIM and NeMo Guardrails](https://developer.nvidia.com/blog/secure-generative-ai-with-nvidia-nim-and-nemo-guardrails/) | NIM integration |
| ⭐⭐ | [NeMo Guardrails and LangChain Templates](https://developer.nvidia.com/blog/nemo-guardrails-and-langchain-templates/) | LangChain integration |
| ⭐⭐ | [Prevent LLM Hallucinations with Cleanlab](https://developer.nvidia.com/blog/prevent-llm-hallucinations-with-the-cleanlab-trustworthy-language-model/) | Hallucination prevention |
| ⭐⭐ | [Measuring Effectiveness of AI Guardrails](https://developer.nvidia.com/blog/measuring-the-effectiveness-and-performance-of-ai-guardrails/) | Evaluation metrics |
| ⭐ | [Stream Smarter and Safer](https://developer.nvidia.com/blog/stream-smarter-and-safer-learn-how-nvidia-nemo-guardrails-enhance-llm-output-streaming/) | Streaming |
| ⭐ | [Mitigating Stored Prompt Injection Attacks](https://developer.nvidia.com/blog/mitigating-stored-prompt-injection-attacks-against-llm-applications/) | Security |

### Videos

| Title | Duration | Topics |
|-------|----------|--------|
| [Customizing AI Agents for Tool Calling](https://developer.nvidia.com/video/customizing-ai-agents-tool-calling) | ~30 min | Tool use, NeMo microservices |
| [Optimize AI Agents With a Data Flywheel](https://developer.nvidia.com/video/optimize-ai-agents-data-flywheel) | ~20 min | Data flywheel pattern |
| [Beyond the Algorithm with NVIDIA](https://developer.nvidia.com/video/beyond-algorithm-nvidia) | ~45 min | NeMo ecosystem overview |

### NIM Models to Know

| Model | NIM ID | Purpose |
|-------|--------|---------|
| NemoGuard JailbreakDetect | `nvidia/nemoguard-jailbreak-detect` | Jailbreak detection |
| NemoGuard ContentSafety | `nvidia/llama-3.1-nemoguard-8b-content-safety` | Content moderation |
| NemoGuard TopicControl | `nvidia/llama-3.1-nemoguard-8b-topic-control` | Topic classification |

---

## 🧪 Hands-On Labs

### Lab 1: Basic Guardrails Setup
```bash
# Install
pip install nemoguardrails>=0.19.0

# Create config directory
mkdir -p config

# Create config.yml
cat > config/config.yml << 'EOF'
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo
EOF

# Create rails.co
cat > config/rails.co << 'EOF'
import core
import llm

flow main
    activate greeting
    activate llm_continuation

flow greeting
    user said "hi" or user said "hello"
    bot say "Hello! How can I help you?"

flow llm_continuation
    user said something unexpected
    $response = await GenerateResponseAction
    bot say $response
EOF

# Test
nemoguardrails chat --config config/
```

### Lab 2: Add Input Safety Rail
```colang
# Add to rails.co
flow check_jailbreak
    user said $text
    if "ignore" in $text.lower() and "instruction" in $text.lower()
        bot say "I can't modify my core behavior."
        abort
```

### Lab 3: Add Output Fact-Checking
```colang
flow fact_check_output
    bot said $text
    $context = $system.context
    $is_grounded = await check_facts(response=$text, context=$context)
    if not $is_grounded
        bot say "I'm not certain about that information."
        abort
```

### Lab 4: RAG with Retrieval Rails
```colang
flow rag_with_guardrails
    user said $query
    $chunks = await RetrieveAction(query=$query)
    $relevant = await filter_by_relevance(chunks=$chunks, threshold=0.7)
    if len($relevant) == 0
        bot say "I don't have information on that topic."
        abort
    $response = await GenerateWithContextAction(context=$relevant)
    bot say $response
```

---

## ✅ Study Checklist

### Core Concepts
- [ ] Understand the 5 types of guardrails (input, dialog, retrieval, execution, output)
- [ ] Know the guardrails architecture and flow
- [ ] Understand event-driven interaction model

### Colang 2.0
- [ ] Write basic flows with `flow main` entry point
- [ ] Use `activate` / `deactivate` for flow control
- [ ] Use `await` for blocking actions
- [ ] Use `start` for non-blocking actions
- [ ] Understand variable scoping (local by default)
- [ ] Know the import system (`import core`, `import llm`)

### Integration
- [ ] LangChain integration with `RunnableRails`
- [ ] LangGraph multi-agent integration
- [ ] Streaming support

### NVIDIA NIMs
- [ ] Know the NemoGuard models (Jailbreak, ContentSafety, TopicControl)
- [ ] Configure NIMs in guardrails config

### Agentic AI Safety
- [ ] Tool use validation patterns
- [ ] Multi-agent guardrails
- [ ] Data flywheel for continuous improvement

### Hands-On
- [ ] Complete Lab 1: Basic setup
- [ ] Complete Lab 2: Input rails
- [ ] Complete Lab 3: Output rails
- [ ] Complete Lab 4: RAG rails
- [ ] Test your Woodcreek implementation

---

## 🎯 Exam Tips

1. **Focus on Colang 2.0** - The exam likely tests the latest syntax
2. **Know the 5 rail types** - Be able to identify which rail type applies to a scenario
3. **Understand async patterns** - `await` vs `start` is important
4. **NIM integration** - Know how to configure NemoGuard models
5. **Agentic patterns** - Tool safety, multi-agent coordination
6. **Evaluation metrics** - How to measure guardrail effectiveness

---

## 📅 Study Schedule (6 Days)

| Day | Focus | Time |
|-----|-------|------|
| 1 | Core concepts + Colang 2.0 basics | 2-3 hrs |
| 2 | All 5 rail types + hands-on labs | 3-4 hrs |
| 3 | Integration patterns (LangChain/LangGraph) | 2-3 hrs |
| 4 | NIMs + Agentic safety | 2-3 hrs |
| 5 | Review blog posts + videos | 2-3 hrs |
| 6 | Practice + rest | 1-2 hrs |

---

**Good luck on your exam! 🍀**

*Last updated: December 29, 2024*