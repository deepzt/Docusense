import os

from rag_core import (
    build_vector_store_for_file,
    get_openai_model,
    get_ollama_model,
    build_context_and_answer,
)


def main() -> None:
    print("File QA CLI (CSV / Excel / PDF)")
    print("=" * 60)

    file_path = input("Enter path to CSV/XLS/XLSX/PDF file: ").strip()
    if not file_path:
        print("No file path provided. Exiting.")
        return

    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        vector_store, info = build_vector_store_for_file(file_path)
        print(info)
    except Exception as e:
        print(f"Error building vector store: {e}")
        return

    openai_model, openai_model_name = get_openai_model()
    ollama_model, ollama_model_name = get_ollama_model()

    if openai_model_name:
        print(f"Using OpenAI model: {openai_model_name}")
    elif ollama_model_name:
        print(f"Using Ollama model: {ollama_model_name}")
    else:
        print("No OpenAI or Ollama model available. Answers will be based on retrieved context only.")

    print("\nYou can now ask questions about the file.")
    print("Type 'quit', 'exit', or 'q' to stop.\n")

    history: list[tuple[str, str]] = []

    while True:
        try:
            question = input("Your question: ").strip()
            if question.lower() in {"quit", "exit", "q"}:
                print("\nGoodbye!")
                break
            if not question:
                continue

            answer = build_context_and_answer(
                vector_store,
                question,
                openai_model,
                openai_model_name,
                ollama_model,
                ollama_model_name,
                history,
            )
            history.append((question, answer))
            if len(history) > 20:
                history = history[-20:]
            print("\nAnswer:\n")
            print(answer)
            print("\n" + "-" * 60 + "\n")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            print("Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    main()
