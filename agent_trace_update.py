# Updated run() method for agent.py
# This creates the hierarchical trace structure shown in the screenshot

def run(self, question: str) -> Dict[str, Any]:
    """
    Run the complete research workflow with proper Galileo session/trace hierarchy.

    Creates structure:
    Session
      └── trace
          └── Chain (plan)
              └── Chat Model
          └── Chain (search)
              └── Chat Model
          └── Chain (curate)
          └── Chain (analyze)
              └── Chat Model
          └── Chain (synthesize)
              └── Chat Model
          └── Chain (validate)

    Args:
        question: Research question

    Returns:
        Dictionary with answer, metrics, and trace info
    """
    from galileo import galileo_context
    from galileo.handlers.langchain import GalileoCallback
    from langchain_core.runnables import RunnableConfig

    print("=" * 80)
    print(f"🔍 RESEARCH QUESTION: {question}")
    print("=" * 80)

    # 1. Start a Galileo session (top level in hierarchy)
    session_name = f"Research: {question[:60]}"
    session_id = f"research-{int(time.time())}"

    galileo_context.start_session(
        name=session_name,
        external_id=session_id
    )
    print(f"✓ Started Galileo session: {session_name}")

    # 2. Initialize state
    initial_state = {
        "question": question,
        "plan": "",
        "search_results": [],
        "curated_sources": [],
        "insights": "",
        "final_answer": "",
        "step_metrics": []
    }

    # 3. Create callback (creates trace under session)
    galileo_callback = GalileoCallback(
        galileo_logger=self.galileo_logger,
        start_new_trace=True,      # Creates new trace under session
        flush_on_chain_end=True,   # Auto-flush when done
    )

    # 4. Create config with metadata
    config = RunnableConfig(
        callbacks=[galileo_callback],
        run_name="Research Pipeline",  # Shows as trace name
        tags=["research", "multi-step", "langgraph"],
        metadata={
            "question": question,
            "num_steps": 6,
            "model": self.model,
            "project": self.project,
            "log_stream": self.log_stream
        }
    )

    # 5. Run the graph (each node shows as Chain)
    try:
        final_state = self.app.invoke(initial_state, config=config)
        print("\n✓ Research pipeline completed")
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        # End session on error too
        galileo_context.end_session()
        raise

    # 6. Extract trace information
    trace_id = self.galileo_logger.trace_id
    trace_url = self._build_trace_url(trace_id)

    if trace_id:
        print(f"✓ Trace ID: {trace_id}")
        print(f"✓ View trace: {trace_url}")

    # 7. End the session (closes the top-level container)
    galileo_context.end_session()
    print("✓ Galileo session ended")

    # 8. Flush evaluator (separate context)
    self.evaluator.flush()
    print("✓ Evaluator traces flushed")

    # 9. Return results with trace info
    return {
        "question": question,
        "answer": final_state["final_answer"],
        "plan": final_state["plan"],
        "insights": final_state["insights"],
        "metrics": final_state["step_metrics"],
        "sources": final_state.get("curated_sources") or final_state.get("search_results", []),
        "trace_id": trace_id,
        "trace_url": trace_url,
        "session_id": session_id,
        "session_name": session_name
    }
