"""Entry point: give the builder a request, watch it spec -> code -> test, see the verdict.

    OLLAMA_MODEL=llama3.2:latest python main.py
"""

from src.graph import build_graph

def print_result(result):
    """Normal end-of-build output (on success)."""
    print("\n=== spec ===\n" + result["spec"])
    print("\n=== code ===\n" + result["code"])
    print("\n=== tests ===\n" + result["tests"])
    print("\n=== status:", result["status"])

def print_stuck_report(result):
    """Gave up: replay the ledger so the user (the judge) sees the WHOLE trajectory -
    each attempt's code and how the sandbox judged it."""
    print("\n" + "=" * 60)
    print("STUCK - gave up after the retry brake. What was tried:")
    print("=" * 60)
    attempt = 0
    for e in result["ledger"]:
        if e.artifact == "spec":
            print(f"\n--- spec (orchestrator) ---\n{e.content}")
        elif e.artifact == "tests":
            print(f"\n--- tests (qa) ---\n{e.content}")
        elif e.artifact == "code":
            attempt += 1
            print(f"\n--- coder attempt {attempt} ---\n{e.content}")
        elif e.artifact == "result":
            print(f"\n--- sandbox verdict ---\n{e.content[:800]}")

def run_one_build(app):
    """Drive ONE build task to success or user-abandon. Returns when this build ends."""
    base_request = input("\nWhat function should I build? ").strip()
    if not base_request:
        base_request = "write a function is_palindrome(s) that checks if a string is a palindrome"
        print(f"(no input - using default: {base_request})")

    feedbacks = []                                   # resets per build (it's a local)
    while True:                                      # the intervention loop (one build)
        request = base_request
        if feedbacks:
            request += ("\n\nFeedback from previous failed attempts (address ALL of these):\n"
                        + "\n".join(f"- {f}" for f in feedbacks))

        result = app.invoke({"request": request})

        if result["status"] == "ok":
            print_result(result)
            print("\nBuild succeeded.")
            return

        print_stuck_report(result)
        print("\nType FEEDBACK to guide the next attempt, ENTER to retry as-is, or 'q' to abandon.")
        fb = input("> ").strip()
        if fb.lower() == "q":
            print("Build abandoned.")
            return
        if fb:
            feedbacks.append(fb)


def main():
    app = build_graph()                              # built ONCE, reused for every build
    while True:                                      # the session: many builds
        run_one_build(app)
        again = input("\nBuild another? [y/N]: ").strip().lower()
        if again != "y":
            print("Done.")
            break



if __name__ == "__main__":
    main()
