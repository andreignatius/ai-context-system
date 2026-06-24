"""Entry point: give the builder a request, watch it spec -> code -> test, see the verdict.

    OLLAMA_MODEL=llama3.2:latest python main.py
"""

from src.graph import build_graph


def main():
    app = build_graph()
    request = input("What function should I build? ").strip()
    if not request:                       # fallback for quick testing
        request = "write a function is_palindrome(s) that checks if a string is a palindrome"
        print(f"(no input - using default: {request})")

    result = app.invoke({"request": request})

    print("\n=== spec ===\n" + result["spec"])
    print("\n=== code ===\n" + result["code"])
    print("\n=== tests ===\n" + result["tests"])
    print("\n=== status:", result["status"])
    print("=== tests passed:", result["test_result"]["passed"])
    if not result["test_result"]["passed"]:
        print(result["test_result"]["failures"][:1500])


if __name__ == "__main__":
    main()
