"""Entry point: give the builder a request, watch it spec -> code -> test, see the verdict.

    OLLAMA_MODEL=llama3.2:latest python main.py
"""

from src.graph import build_graph
from src.config import get_langfuse_handler

_handler = get_langfuse_handler()

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
    base_request = input("\nWhat function should I build? ").strip()
    if not base_request:
        base_request = "write a function is_palindrome(s) that checks if a string is a palindrome"
        print(f"(no input - using default: {base_request})")

    build = {"request": base_request, "feedback": "", "fix_target": ""}   # a FRESH build
    while True:
        result = app.invoke(build, config={"callbacks": [_handler]})

        if result["status"] == "ok":
            print_result(result)
            print("\nBuild succeeded.")
            return

        print_stuck_report(result)
        print("\nFix which artifact? [spec / tests / code]  (or 'q' to abandon)")
        target = input("> ").strip().lower()
        if target == "q":
            print("Build abandoned.")
            return
        if target not in ("spec", "tests", "code"):
            print("(unrecognised - retrying as a fresh build)")
            target = ""
        fb = input("Feedback for this fix: ").strip()

        # MEMORY: carry the prior build forward.   TARGETING: edit only `target`.
        build = {
            "request": base_request,
            "spec": result["spec"],
            "tests": result["tests"],
            "code": result["code"],
            "test_result": result["test_result"],
            "feedback": fb,
            "fix_target": target,
        }



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
