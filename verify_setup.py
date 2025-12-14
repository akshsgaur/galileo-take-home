"""
Setup verification script.
Run this to check if all dependencies are installed and API keys are configured.
"""

import sys
import os


def check_imports():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    print("-" * 60)

    required_packages = [
        ("langchain", "LangChain"),
        ("langchain_openai", "LangChain OpenAI"),
        ("langgraph", "LangGraph"),
        ("galileo", "Galileo SDK"),
        ("openai", "OpenAI"),
        ("dotenv", "Python Dotenv"),
        ("tavily", "Tavily Python"),
    ]

    missing = []

    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            missing.append(package)

    print("-" * 60)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def check_env():
    """Check if environment variables are configured."""
    print("\nChecking environment variables...")
    print("-" * 60)

    from dotenv import load_dotenv
    load_dotenv()

    required_vars = {
        "GALILEO_API_KEY": "Galileo API Key",
        "OPENAI_API_KEY": "OpenAI API Key",
        "TAVILY_API_KEY": "Tavily API Key",
    }

    missing = []

    for var, name in required_vars.items():
        value = os.getenv(var)
        if value and value != f"your-{var.lower().replace('_', '-')}-here":
            print(f"✓ {name}")
        else:
            print(f"✗ {name} - NOT SET")
            missing.append(var)

    print("-" * 60)

    if missing:
        print(f"\n❌ Missing environment variables: {', '.join(missing)}")
        print("\nCreate .env file from .env.example and add your API keys:")
        print("  cp .env.example .env")
        print("  # Then edit .env with your actual API keys")
        return False
    else:
        print("\n✓ All environment variables configured!")
        return True


def check_galileo_connection():
    """Test Galileo connection."""
    print("\nTesting Galileo connection...")
    print("-" * 60)

    try:
        from galileo import galileo_context
        from galileo.openai import openai

        galileo_context.init(
            project="setup-verification",
            log_stream="connection-test"
        )

        print("✓ Galileo initialized successfully")
        return True

    except Exception as e:
        print(f"✗ Galileo initialization failed: {e}")
        print("\nCheck your GALILEO_API_KEY in .env file")
        return False


def check_openai_connection():
    """Test OpenAI connection."""
    print("\nTesting OpenAI connection...")
    print("-" * 60)

    try:
        from galileo.openai import openai
        import os

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Simple test call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=5
        )

        if response.choices[0].message.content:
            print("✓ OpenAI connection successful")
            return True
        else:
            print("✗ OpenAI returned empty response")
            return False

    except Exception as e:
        print(f"✗ OpenAI connection failed: {e}")
        print("\nCheck your OPENAI_API_KEY in .env file")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("RESEARCH AGENT SETUP VERIFICATION")
    print("=" * 60)

    results = []

    # Check imports
    results.append(check_imports())

    # Check environment variables
    if results[-1]:  # Only if imports succeeded
        results.append(check_env())

    # Check connections (only if env vars are set)
    if all(results):
        results.append(check_galileo_connection())
        results.append(check_openai_connection())

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    if all(results):
        print("\n✅ All checks passed! You're ready to run the agent.")
        print("\nNext steps:")
        print("  python agent.py          # Run single question")
        print("  python evaluate.py       # Run full evaluation")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
