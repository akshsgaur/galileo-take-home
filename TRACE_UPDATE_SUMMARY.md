# Trace Structure Update Summary

## Changes Made

Updated the research agent to create the proper **Session → trace → Chain → Chat Model** hierarchy in Galileo traces, matching the structure shown in the screenshot.

---

## Files Modified

### 1. `agent.py` - `run()` method

**Key changes**:

```python
# BEFORE:
galileo_callback = GalileoCallback(...)
final_state = self.app.invoke(initial_state, config={"callbacks": [callback]})

# AFTER:
from galileo import galileo_context
from langchain_core.runnables import RunnableConfig

# Start session (creates "Session" in hierarchy)
galileo_context.start_session(
    name=f"Research: {question[:60]}",
    external_id=f"research-{timestamp}"
)

# Create callback (creates "trace" under Session)
galileo_callback = GalileoCallback(
    galileo_logger=self.galileo_logger,
    start_new_trace=True,
    flush_on_chain_end=True
)

# Create config with metadata
config = RunnableConfig(
    callbacks=[galileo_callback],
    run_name="Research Pipeline",  # Trace name
    tags=["research", "multi-step", "langgraph"],
    metadata={...}
)

# Invoke graph
final_state = self.app.invoke(initial_state, config=config)

# End session
galileo_context.end_session()
```

**Added return fields**:
- `session_id`: Unique session identifier
- `session_name`: Human-readable session name

### 2. `api_server.py` - Response model

**Added fields**:
```python
class ResearchResponse(BaseModel):
    # ... existing fields
    trace_id: Optional[str] = None
    trace_url: Optional[str] = None
    session_id: Optional[str] = None      # ← NEW
    session_name: Optional[str] = None    # ← NEW
```

---

## Expected Trace Structure

### In Galileo Console

```
📦 Session: "Research: What are the latest trends in AI observability?"
  └── 🔗 trace: "Research Pipeline"
      ├── ⛓️ Chain: "plan"
      │   └── 🤖 Chat Model: gpt-4o-mini
      ├── ⛓️ Chain: "search"
      │   └── 🤖 Chat Model: gpt-4o-mini
      ├── ⛓️ Chain: "curate"
      ├── ⛓️ Chain: "analyze"
      │   └── 🤖 Chat Model: gpt-4o-mini
      ├── ⛓️ Chain: "synthesize"
      │   └── 🤖 Chat Model: gpt-4o-mini
      └── ⛓️ Chain: "validate"
```

### Hierarchy Breakdown

| Level | Type | Name | Description |
|-------|------|------|-------------|
| 1 | Session | "Research: {question}" | Top-level container for all traces |
| 2 | trace | "Research Pipeline" | Single trace representing the graph execution |
| 3 | Chain | Node names (plan, search, etc.) | Each LangGraph node |
| 4 | Chat Model | gpt-4o-mini | LLM calls within nodes |

---

## Expected Console Output

```bash
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
✓ View trace: https://app.galileo.ai/?project=research-agent-web&logStream=web-interface&traceId=a1b2c3d4...
✓ Galileo session ended
✓ Evaluator traces flushed
```

---

## API Response Example

```json
{
  "question": "What are the latest trends in AI observability?",
  "answer": "# Latest Trends in AI Observability (2024)\n\nAI observability...",
  "plan": "**Core Focus Areas:**\n1. Current state...",
  "insights": "**Key Findings:**\n1. Observability platforms...",
  "metrics": [
    {
      "step": "plan",
      "latency": 4.22,
      "quality_score": 8,
      "reasoning": "Comprehensive plan with specific queries"
    },
    // ... more metrics
  ],
  "sources": [
    {
      "title": "AI Observability Trends 2024",
      "url": "https://example.com/article",
      "snippet": "...",
      "domain": "example.com",
      "confidence": 0.87,
      "source_type": "industry"
    },
    // ... more sources
  ],
  "trace_id": "a1b2c3d4-5678-90ef-ghij-klmnopqrstuv",
  "trace_url": "https://app.galileo.ai/?project=research-agent-web&logStream=web-interface&traceId=a1b2c3d4...",
  "session_id": "research-1734210567",
  "session_name": "Research: What are the latest trends in AI observability?"
}
```

---

## Testing

### 1. Restart Backend Server

```bash
cd /Users/akshitgaur/hackathon/galileo-tech-stack/research-agent
python api_server.py
```

**Expected startup output**:
```
✓ Galileo API key found (ending in ...abc12345)
✓ Initializing Galileo: project='research-agent-web', log_stream='web-interface'
✓ Galileo logger ready
✓ Using model: gpt-4o-mini
✓ Evaluator initialized with log stream: web-interface-eval
✓ LangGraph workflow compiled

================================================================================
🚀 Starting Research Agent API Server
================================================================================
Server will be available at: http://localhost:8000
API docs will be available at: http://localhost:8000/docs
Ready to accept requests from Next.js frontend
================================================================================
```

### 2. Submit Test Query

Use the frontend or submit directly:

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest trends in AI observability?"}'
```

### 3. Verify in Galileo Console

1. Go to https://app.galileo.ai/
2. Select project: `research-agent-web`
3. Select log stream: `web-interface`
4. Look for the session in the Messages view
5. Click to expand: **Session → trace → Chain nodes**

**You should see**:
- ✅ Session named "Research: What are the latest trends..."
- ✅ Nested trace named "Research Pipeline"
- ✅ 6 Chain entries (plan, search, curate, analyze, synthesize, validate)
- ✅ Chat Model entries under relevant chains
- ✅ Proper nesting with visual hierarchy
- ✅ Tags: research, multi-step, langgraph
- ✅ Metadata: question, num_steps, model, project, log_stream

### 4. Verify Trace URL

Click the "View in Galileo" button in the frontend or use the trace_url from the API response. It should:
- ✅ Open directly to the correct trace
- ✅ Show the full hierarchy
- ✅ Display all metadata
- ✅ Show Luna-2 metrics (context_adherence, hallucination, prompt_injection)

---

## Troubleshooting

### Issue: Flat trace structure (no Session)

**Symptom**: All entries at same level, no hierarchy

**Fix**: Ensure `galileo_context.start_session()` is called **before** creating callback

**Check**:
```python
# Verify order in run() method:
galileo_context.start_session(...)  # FIRST
callback = GalileoCallback(...)     # SECOND
graph.invoke(..., config={"callbacks": [callback]})  # THIRD
```

### Issue: Multiple sessions per query

**Symptom**: Each LangGraph node creates a new session

**Fix**: Only call `start_session()` once in `run()`, not in step functions

**Check**: Search for `start_session` in code - should only appear in `run()` method

### Issue: Missing Chain entries

**Symptom**: Some LangGraph nodes don't show up

**Fix**: Ensure callback is passed in config parameter

**Check**:
```python
# ✅ Correct:
config = RunnableConfig(callbacks=[callback])
graph.invoke(state, config=config)

# ✗ Wrong:
graph.invoke(state)  # No callback!
```

### Issue: LLM calls not showing as "Chat Model"

**Symptom**: LLM calls embedded in Chain, not separate

**Solution**: Your current `galileo.openai` wrapper should work. If not, switch to LangChain's `ChatOpenAI`:

```python
from langchain_openai import ChatOpenAI

self.chat_model = ChatOpenAI(model="gpt-4o-mini")

# In step functions:
response = self.chat_model.invoke(messages)  # Shows as Chat Model
```

---

## Key Benefits

1. ✅ **Better organization**: Sessions group related traces
2. ✅ **Clear hierarchy**: Easy to see execution flow
3. ✅ **Better debugging**: Drill down from session → trace → step → LLM call
4. ✅ **Shareable links**: Trace URLs go directly to the right view
5. ✅ **Metadata tracking**: Question, model, project, tags all captured
6. ✅ **Luna-2 metrics**: Automatic scoring on all LLM calls

---

## Next Steps

1. **Test the changes**: Run a query and verify trace structure
2. **Check Luna-2 metrics**: Ensure context_adherence, hallucination scores appear
3. **Verify trace URL**: Click link in frontend, should go directly to trace
4. **Monitor sessions**: Multiple queries should create multiple sessions
5. **Review metadata**: Check that all tags and metadata are captured

---

## Reference Files Created

1. `agent_trace_update.py` - Example `run()` method implementation
2. `TRACE_STRUCTURE_GUIDE.md` - Complete guide with examples and troubleshooting
3. `TRACE_UPDATE_SUMMARY.md` - This file

---

**Updated**: December 14, 2024
**Status**: Ready to test
**Expected Result**: Hierarchical trace matching screenshot
