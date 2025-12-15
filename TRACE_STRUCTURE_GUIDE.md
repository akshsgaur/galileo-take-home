# Galileo Trace Structure Guide

This guide explains how to achieve the hierarchical trace structure shown in the Galileo UI.

## Target Structure

```
Session: "Research: What are the latest trends..."
└── trace: "Research Pipeline"
    ├── Chain: "plan"
    │   └── Chat Model: gpt-4o-mini
    ├── Chain: "search"
    │   └── Chat Model: gpt-4o-mini (query extraction)
    ├── Chain: "curate"
    ├── Chain: "analyze"
    │   └── Chat Model: gpt-4o-mini
    ├── Chain: "synthesize"
    │   └── Chat Model: gpt-4o-mini
    └── Chain: "validate"
```

---

## Implementation Steps

### 1. Start a Galileo Session

Sessions are the top-level containers for traces.

```python
from galileo import galileo_context

# Start session (shows as "Session" in UI)
galileo_context.start_session(
    name=f"Research: {question[:60]}",
    external_id=f"research-{timestamp}"
)
```

### 2. Create GalileoCallback with Proper Flags

```python
from galileo.handlers.langchain import GalileoCallback

galileo_callback = GalileoCallback(
    galileo_logger=self.galileo_logger,
    start_new_trace=True,      # Creates "trace" under Session
    flush_on_chain_end=True,   # Auto-flush when graph completes
)
```

### 3. Invoke LangGraph with Config

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    callbacks=[galileo_callback],
    run_name="Research Pipeline",  # Shows as trace name
    tags=["research", "multi-step"],
    metadata={"question": question}
)

final_state = self.app.invoke(initial_state, config=config)
```

### 4. End Session After Completion

```python
galileo_context.end_session()
```

---

## LangGraph Node Naming

Each node in your StateGraph automatically shows as a **Chain** entry with the node name:

```python
workflow.add_node("plan", self.plan_step)          # Shows as: Chain: "plan"
workflow.add_node("search", self.search_step)      # Shows as: Chain: "search"
workflow.add_node("curate", self.curate_step)      # Shows as: Chain: "curate"
workflow.add_node("analyze", self.analyze_step)    # Shows as: Chain: "analyze"
workflow.add_node("synthesize", self.synthesize_step)  # Shows as: Chain: "synthesize"
workflow.add_node("validate", self.validate_step)  # Shows as: Chain: "validate"
```

---

## LLM Calls as "Chat Model" Entries

### Option 1: Use Galileo-wrapped OpenAI client (Current Approach)

Your current approach using `galileo.openai` works:

```python
from galileo.openai import openai

self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In step functions:
response = self.client.chat.completions.create(
    model=self.model,
    messages=[...],
    temperature=0.7
)
```

This **should** show as "Chat Model: gpt-4o-mini" nested under the Chain.

### Option 2: Use LangChain ChatOpenAI (Guaranteed to show)

For guaranteed "Chat Model" entries, use LangChain's wrapper:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Create in __init__ or per-call
chat = ChatOpenAI(
    model=self.model,
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# In step functions:
messages = [
    SystemMessage(content="You are an expert..."),
    HumanMessage(content=prompt)
]

response = chat.invoke(messages)  # Shows as "Chat Model" in trace
plan = response.content
```

### Option 3: Hybrid Approach (Recommended)

Keep your current `galileo.openai` client but add LangChain wrapper for tracing:

```python
class ResearchAgent:
    def __init__(self, ...):
        # Keep galileo.openai for direct calls
        from galileo.openai import openai
        self.client = openai.OpenAI(...)

        # Add LangChain wrapper for traced calls
        from langchain_openai import ChatOpenAI
        self.chat_model = ChatOpenAI(
            model=self.model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def plan_step(self, state: AgentState) -> AgentState:
        # Use chat_model instead of client for traced calls
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content="You are an expert research strategist."),
            HumanMessage(content=prompt)
        ]

        response = self.chat_model.invoke(messages)  # Traced as "Chat Model"
        plan = response.content
```

---

## Debugging Trace Structure

### Check Session Status

```python
# After start_session()
logger = galileo_context.get_logger_instance()
print(f"Session active: {logger.current_parent() is not None}")
```

### Check Trace ID

```python
# After graph invocation
trace_id = self.galileo_logger.trace_id
print(f"Trace ID: {trace_id}")
```

### View in Galileo Console

```python
trace_url = f"{console_url}?project={project}&logStream={log_stream}&traceId={trace_id}"
print(f"View trace: {trace_url}")
```

---

## Common Issues

### Issue 1: Flat structure (no nesting)

**Symptom**: All entries at same level, no Session → trace → Chain hierarchy

**Fix**: Ensure you call `galileo_context.start_session()` **before** creating callback

```python
# ✓ Correct order:
galileo_context.start_session(...)
callback = GalileoCallback(...)
graph.invoke(state, config={"callbacks": [callback]})

# ✗ Wrong order:
callback = GalileoCallback(...)
galileo_context.start_session(...)  # Too late!
```

### Issue 2: LLM calls not showing as "Chat Model"

**Symptom**: LLM calls embedded in Chain, not separate entries

**Fix**: Use LangChain's `ChatOpenAI` instead of raw OpenAI client

```python
# ✗ Won't show as separate Chat Model:
response = openai_client.chat.completions.create(...)

# ✓ Will show as Chat Model:
response = chat_model.invoke(messages)
```

### Issue 3: Multiple traces instead of single session

**Symptom**: Each run creates separate trace, no session grouping

**Fix**: Don't call `start_session()` multiple times. One session per research query:

```python
# ✓ Correct:
def run(self, question):
    galileo_context.start_session(...)
    # ... do work ...
    galileo_context.end_session()

# ✗ Wrong - creates multiple sessions:
def plan_step(self, state):
    galileo_context.start_session(...)  # Don't do this per step!
```

### Issue 4: Callback not capturing all nodes

**Symptom**: Some LangGraph nodes missing from trace

**Fix**: Ensure callback is passed in `config` parameter of `invoke()`:

```python
# ✓ Correct:
config = RunnableConfig(callbacks=[callback])
graph.invoke(state, config=config)

# ✗ Wrong:
graph.invoke(state)  # No callback!
```

---

## Complete Example

```python
from galileo import galileo_context
from galileo.handlers.langchain import GalileoCallback
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

class ResearchAgent:
    def __init__(self, project, log_stream):
        self.galileo_logger = GalileoLogger(project=project, log_stream=log_stream)
        self.chat_model = ChatOpenAI(model="gpt-4o-mini")
        self.app = self._build_graph()

    def run(self, question: str):
        # 1. Start session
        galileo_context.start_session(
            name=f"Research: {question[:60]}",
            external_id=f"research-{int(time.time())}"
        )

        # 2. Create callback
        callback = GalileoCallback(
            galileo_logger=self.galileo_logger,
            start_new_trace=True,
            flush_on_chain_end=True
        )

        # 3. Create config
        config = RunnableConfig(
            callbacks=[callback],
            run_name="Research Pipeline"
        )

        # 4. Invoke graph
        state = {"question": question, ...}
        final_state = self.app.invoke(state, config=config)

        # 5. Get trace info
        trace_id = self.galileo_logger.trace_id
        trace_url = self._build_trace_url(trace_id)

        # 6. End session
        galileo_context.end_session()

        return {..., "trace_id": trace_id, "trace_url": trace_url}

    def plan_step(self, state):
        # Use chat_model for traced LLM calls
        messages = [
            SystemMessage(content="You are an expert..."),
            HumanMessage(content=f"Create a plan for: {state['question']}")
        ]

        response = self.chat_model.invoke(messages)  # Shows as "Chat Model"
        return {..., "plan": response.content}
```

---

## Expected Console Output

```
✓ Galileo API key found (ending in ...abc12345)
✓ Initializing Galileo: project='research-agent-web', log_stream='web-interface'
✓ Galileo logger ready
✓ Using model: gpt-4o-mini
✓ Evaluator initialized with log stream: web-interface-eval
✓ LangGraph workflow compiled

================================================================================
🔍 RESEARCH QUESTION: What are the latest trends in AI observability?
================================================================================
✓ Started Galileo session: Research: What are the latest trends in AI observability?

🧠 STEP 1: Planning...
✓ Plan generated (4.22s, quality: 8/10)

🔍 STEP 2: Searching...
✓ Found 12 results from 3 queries (3.45s, relevance: 9/10)

🗂️  STEP 3: Curating sources...
✓ Curated 8 sources (85% avg confidence, 0.23s)

📊 STEP 4: Analyzing...
✓ Insights extracted (7.21s, completeness: 9/10)

✨ STEP 5: Synthesizing...
✓ Answer synthesized (5.43s, quality: 9/10)

🛡️  STEP 6: Validating answer...
✓ Answer validated (2.14s, grounded: 9/10)

✓ Research pipeline completed
✓ Trace ID: a1b2c3d4-5678-90ef-ghij-klmnopqrstuv
✓ View trace: https://console.getgalileo.io/...
✓ Galileo session ended
✓ Evaluator traces flushed
```

---

## Verification Checklist

- [ ] Session starts before callback creation
- [ ] Callback has `start_new_trace=True`
- [ ] Callback passed in `config` parameter
- [ ] LLM calls use LangChain `ChatOpenAI` (or galileo.openai wrapper)
- [ ] Session ends after graph completion
- [ ] Trace URL is generated and returned

---

## References

- [LangChain Integration Docs](https://v2docs.galileo.ai/sdk-api/third-party-integrations/langchain/langchain)
- [Running Experiments](https://v2docs.galileo.ai/sdk-api/third-party-integrations/langchain/experiments)
- [Galileo Protect](https://v2docs.galileo.ai/sdk-api/third-party-integrations/langchain/protect)
