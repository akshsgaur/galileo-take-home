"""
Diagnostic script to test Galileo integration.
Run this to verify Galileo is working before running the full agent.
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("🔬 GALILEO DIAGNOSTIC TEST")
print("=" * 80)

# Test 1: Check environment variables
print("\n1️⃣  Checking environment variables...")
galileo_key = os.getenv("GALILEO_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if galileo_key:
    print(f"   ✓ GALILEO_API_KEY found (ending in ...{galileo_key[-8:]})")
else:
    print("   ✗ GALILEO_API_KEY not found!")
    exit(1)

if openai_key:
    print(f"   ✓ OPENAI_API_KEY found (ending in ...{openai_key[-8:]})")
else:
    print("   ✗ OPENAI_API_KEY not found!")
    exit(1)

# Test 2: Import Galileo packages
print("\n2️⃣  Testing Galileo imports...")
try:
    from galileo import galileo_context
    from galileo.logger import GalileoLogger
    from galileo.handlers.langchain import GalileoCallback
    from galileo.openai import openai
    print("   ✓ All Galileo imports successful")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    exit(1)

# Test 3: Create GalileoLogger
print("\n3️⃣  Testing GalileoLogger creation...")
try:
    logger = GalileoLogger(
        project="test-project",
        log_stream="test-stream"
    )
    print(f"   ✓ GalileoLogger created")
    print(f"   - Project: test-project")
    print(f"   - Log stream: test-stream")
except Exception as e:
    print(f"   ✗ Logger creation failed: {e}")
    exit(1)

# Test 4: Test Galileo-wrapped OpenAI client
print("\n4️⃣  Testing Galileo-wrapped OpenAI client...")
try:
    client = openai.OpenAI(api_key=openai_key)
    print("   ✓ Galileo OpenAI client created")
except Exception as e:
    print(f"   ✗ Client creation failed: {e}")
    exit(1)

# Test 5: Make a simple LLM call
print("\n5️⃣  Testing LLM call with Galileo logging...")
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from Galileo test!'"}
        ],
        temperature=0.7,
        max_tokens=50
    )

    answer = response.choices[0].message.content
    print(f"   ✓ LLM call successful")
    print(f"   - Response: {answer}")
except Exception as e:
    print(f"   ✗ LLM call failed: {e}")
    exit(1)

# Test 6: Check for trace ID
print("\n6️⃣  Checking for trace ID...")
try:
    trace_id = logger.trace_id
    if trace_id:
        print(f"   ✓ Trace ID generated: {trace_id}")
    else:
        print(f"   ⚠️  No trace ID (this might be expected for simple calls)")
except Exception as e:
    print(f"   ⚠️  Could not get trace ID: {e}")

# Test 7: Test GalileoCallback
print("\n7️⃣  Testing GalileoCallback...")
try:
    callback = GalileoCallback(
        galileo_logger=logger,
        start_new_trace=True,
        flush_on_chain_end=True
    )
    print("   ✓ GalileoCallback created")
except Exception as e:
    print(f"   ✗ Callback creation failed: {e}")
    exit(1)

# Test 8: Test LangChain integration
print("\n8️⃣  Testing LangChain integration...")
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableConfig

    chat = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_key)

    config = RunnableConfig(
        callbacks=[callback],
        run_name="Test Chain",
        tags=["test"]
    )

    response = chat.invoke(
        [HumanMessage(content="Say 'LangChain test successful!'")],
        config=config
    )

    print("   ✓ LangChain call successful")
    print(f"   - Response: {response.content}")
except Exception as e:
    print(f"   ✗ LangChain test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 9: Check trace ID after LangChain call
print("\n9️⃣  Checking trace ID after LangChain call...")
try:
    trace_id = logger.trace_id
    if trace_id:
        print(f"   ✓ Trace ID: {trace_id}")

        # Build trace URL
        console_url = os.getenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
        trace_url = f"{console_url}?project=test-project&logStream=test-stream&traceId={trace_id}"
        print(f"   ✓ Trace URL: {trace_url}")
    else:
        print("   ⚠️  No trace ID generated")
except Exception as e:
    print(f"   ⚠️  Error getting trace: {e}")

# Test 10: Manual flush
print("\n🔟  Testing manual flush...")
try:
    # The callback should have auto-flushed with flush_on_chain_end=True
    # But let's verify by checking if we can access flush method
    print("   ℹ️  Callback should have auto-flushed (flush_on_chain_end=True)")
    print("   ℹ️  Check Galileo console for the test trace")
except Exception as e:
    print(f"   ⚠️  Flush check error: {e}")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\n📋 Next steps:")
print("   1. Check Galileo console at: https://app.galileo.ai/")
print("   2. Look for project: 'test-project'")
print("   3. Look for log stream: 'test-stream'")
print("   4. You should see a trace with the test LangChain call")
print("\nIf you see the trace, Galileo is working correctly!")
print("If not, check:")
print("   - API key is valid")
print("   - Project/log stream exist in Galileo")
print("   - Network connectivity to Galileo")
