# Multi-Step Research Agent

AI research agent with step-by-step evaluation using Galileo observability platform.

## Overview

This project implements a multi-step research agent that:
- Takes a research question as input
- Executes 4 sequential steps: Plan → Search → Analyze → Synthesize
- Evaluates each step with custom metrics and LLM-as-judge
- Logs all interactions to Galileo for observability
- Identifies performance bottlenecks across the workflow

## Architecture

```
Question → Plan → Search → Analyze → Synthesize → Answer
            ↓       ↓        ↓          ↓
         Eval    Eval     Eval       Eval
```

### Technology Stack

- **LangGraph**: Workflow orchestration with StateGraph
- **OpenAI GPT-4o-mini**: LLM for planning, analysis, and synthesis
- **Galileo SDK**: Observability and evaluation platform
- **DuckDuckGo**: Web search (no API key required)
- **Python 3.11+**

### Evaluation Framework

| Step | Metrics |
|------|---------|
| **Plan** | Quality Score (1-10, LLM-as-judge) |
| **Search** | Relevance Score (1-10, LLM-as-judge) |
| **Analyze** | Completeness Score (1-10, LLM-as-judge) + Context Adherence (Galileo Luna) |
| **Synthesize** | Answer Quality (1-10, LLM-as-judge) + Context Adherence (Galileo Luna) |

All steps also track **latency** for performance analysis.

## Setup

### 1. Clone and Navigate

```bash
cd research-agent
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```ini
GALILEO_API_KEY=your-galileo-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
```

**Getting API Keys:**
- **Galileo**: Sign up at https://galileo.ai/ and get your API key from the settings
- **OpenAI**: Get your API key from https://platform.openai.com/api-keys

**⚠️ SECURITY WARNING:**
- **NEVER commit your `.env` file to version control**
- The `.gitignore` is configured to protect your API keys
- Only `.env.example` (with placeholder values) should be committed
- See `SECURITY.md` for complete security guidelines

### 5. Verify Setup

Run the verification script to check if everything is configured correctly:

```bash
python verify_setup.py
```

This will check:
- All dependencies are installed
- Environment variables are set
- Galileo connection works
- OpenAI connection works

## Usage

### Run Single Question

Test the agent with a single question:

```bash
python agent.py
```

This runs the default question: *"What are the latest trends in AI observability?"*

### Run Full Evaluation

Evaluate the agent on all 10 test questions:

```bash
python evaluate.py
```

This will:
- Run all questions in `test_questions.py`
- Generate comprehensive metrics
- Identify the bottleneck step (lowest average score)
- Save results to `evaluation_results_<timestamp>.json`

### Quick Test (First 3 Questions)

For faster testing during development:

```bash
python evaluate.py --quick
```

## Project Structure

```
research-agent/
├── .env                  # API keys (create from .env.example)
├── .env.example          # Template for environment variables
├── .gitignore            # Git ignore patterns
├── requirements.txt      # Python dependencies
│
├── agent.py              # Main LangGraph agent
├── tools.py              # Web search function
├── evaluators.py         # LLM-as-judge evaluators
├── test_questions.py     # 10 test questions
├── evaluate.py           # Full evaluation runner
├── verify_setup.py       # Setup verification script
│
└── README.md             # This file
```

## Example Output

```
======================================================================
🔍 RESEARCH QUESTION: What are the latest trends in AI observability?
======================================================================

🧠 STEP 1: Planning...
✓ Plan generated (2.1s, quality: 8/10)

🔍 STEP 2: Searching...
✓ Found 5 results (3.2s, relevance: 9/10)

📊 STEP 3: Analyzing...
✓ Insights extracted (4.5s, completeness: 7/10)

✨ STEP 4: Synthesizing...
✓ Answer synthesized (2.5s, quality: 8/10)

======================================================================
📊 FINAL ANSWER:
======================================================================
The latest trends in AI observability include:

1. **Agentic Evaluation**: New frameworks for evaluating multi-step AI agents
2. **Real-time Guardrails**: Sub-200ms evaluation enabling production blocking
3. **Cost Optimization**: Luna models reducing evaluation costs by 97%
4. **Context Adherence Tracking**: Automated monitoring of RAG pipelines

======================================================================
⏱️  PERFORMANCE SUMMARY:
======================================================================
  plan        2.10s  (quality: 8/10)
  search      3.20s  (relevance: 9/10)
  analyze     4.50s  (completeness: 7/10) ⚠️ BOTTLENECK
  synthesize  2.50s  (quality: 8/10)

  TOTAL      12.30s  (avg score: 8.0/10)

======================================================================
🔗 VIEW DETAILED TRACES IN GALILEO:
======================================================================
https://app.galileo.ai/
======================================================================
```

## Galileo Integration

All LLM calls are automatically logged to Galileo using the OpenAI wrapper:

```python
from galileo.openai import openai
from galileo import galileo_context

# Initialize Galileo
galileo_context.init(
    project="research-agent",
    log_stream="multi-step-research"
)

# Use wrapped OpenAI client (auto-logs all calls)
client = openai.OpenAI()

# ... make LLM calls ...

# Upload traces
galileo_context.flush()
```

### View Traces

After running the agent:
1. Visit https://app.galileo.ai/
2. Navigate to your project: `research-agent`
3. Select log stream: `multi-step-research`
4. View detailed traces with latency, token counts, and Luna metrics

## Test Questions

The evaluation runs on 10 research questions covering AI/ML topics:

1. What are the latest trends in AI observability?
2. How is prompt engineering evolving in 2024?
3. What are the main challenges in deploying LLMs to production?
4. What's new in retrieval-augmented generation (RAG)?
5. How are companies handling AI safety and alignment?
6. What are emerging techniques for evaluating AI systems?
7. How is the AI agent ecosystem developing?
8. What are best practices for LLM monitoring?
9. What's the state of open-source LLMs?
10. How are enterprises adopting generative AI?

## Bottleneck Identification

The evaluation automatically identifies the weakest step:

```
⚠️  Bottleneck Identified: ANALYZE step
   (Lowest avg score: 6.9/10)
```

This helps prioritize improvements (e.g., better prompt engineering for analysis, more sophisticated insight extraction).

## Customization

### Add Your Own Questions

Edit `test_questions.py`:

```python
TEST_QUESTIONS = [
    "Your custom question here",
    # ... more questions
]
```

### Modify Evaluation Criteria

Edit `evaluators.py` to change scoring rubrics in the LLM-as-judge prompts.

### Adjust Search Results

Edit `agent.py`, `search_step()` function:

```python
search_results = search_web(question, num_results=10)  # Get more results
```

## Troubleshooting

### Rate Limiting

If you hit OpenAI rate limits:
- Use `--quick` mode for testing
- Add delays in `evaluate.py` (already includes 1s between questions)

### Search Failures

DuckDuckGo occasionally blocks automated requests:
- The tool includes retry logic and error handling
- Failed searches return empty results with appropriate metrics

### Galileo Connection

If traces don't appear:
- Verify `GALILEO_API_KEY` is correct
- Check project/log stream names match
- Ensure `galileo_context.flush()` is called

## Performance Notes

- **Average time per question**: ~12-15 seconds
- **Token usage**: ~2000-3000 tokens per question
- **Rate limits**: Includes delays to avoid OpenAI rate limiting
- **Caching**: None (each run makes fresh API calls)

## Future Enhancements

Potential improvements:
- [ ] Parallel search queries based on plan
- [ ] Source credibility scoring
- [ ] Citation tracking in final answers
- [ ] Multi-model comparison (GPT-4 vs GPT-4o-mini)
- [ ] Semantic caching for repeated questions
- [ ] Human feedback integration

## Security

**🔒 Protecting Your API Keys**

This project includes comprehensive security measures to prevent API keys from being committed to GitHub:

- **325+ line `.gitignore`** - Blocks all common secret files
- **SECURITY.md** - Complete security guidelines and best practices
- **test_gitignore.sh** - Automated test to verify gitignore works

**Before Committing:**
```bash
# 1. Check what you're committing
git status

# 2. Verify .env is NOT in the list
# 3. Test the gitignore
bash test_gitignore.sh

# 4. Review changes
git diff
```

**What's Protected:**
- `.env` files (all variations)
- API key files (`*api_key*`, `*.key`, etc.)
- Credentials (`credentials.*`, `secrets.*`)
- Results files (may contain API responses)
- Log files (may contain debug info)
- Cloud provider credentials (AWS, GCP, Azure)

**If You Accidentally Commit Secrets:**
1. Immediately rotate the exposed keys
2. Remove from git history (see `SECURITY.md`)
3. Force push to rewrite history

📖 **Read `SECURITY.md` for complete guidelines**

## Resources

- **Galileo Documentation**: https://v2docs.galileo.ai/
- **Galileo Python SDK**: https://github.com/rungalileo/galileo-python
- **SDK Examples**: https://github.com/rungalileo/sdk-examples
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **OpenAI API**: https://platform.openai.com/docs

## License

MIT License - Feel free to use for your own projects.

## Author

Built for Galileo Product Marketing Manager technical assessment.
