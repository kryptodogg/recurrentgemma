import os
import re


def patch_file(filepath):
    try:
        with open(filepath, "r") as f:
            content = f.read()

        original_content = content

        # 1. Strip out all @at.typed decorators
        # Using a regex that handles potential whitespace
        content = re.sub(r"^\s*@at\.typed\s*\n", "", content, flags=re.MULTILINE)

        # 2. Specifically target the array_typing.py to disable the decorator source
        if filepath.endswith("array_typing.py"):
            # Replace the active decorator with a dummy passthrough
            dummy_decorator = """
def typed(function):
    \"\"\"Dummy passthrough for Python 3.12 compatibility.\"\"\"
    return function
"""
            # Replace the jaxtyped import and definition
            content = re.sub(
                r"def typed\(function\).*?return jt\.jaxtyped.*?typechecked\)",
                dummy_decorator.strip(),
                content,
                flags=re.DOTALL,
            )

        if content != original_content:
            with open(filepath, "w") as f:
                f.write(content)
            print(f"Patched: {filepath}")

    except Exception as e:
        print(f"Error patching {filepath}: {e}")


def main():
    print("--- Booting RecurrentGemma AST Patcher (Python 3.12) ---")

    # Path to the specific folder you are using
    base_dir = "/workspace/recurrentgemma/recurrentgemma/jax"

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        print("Please verify the path to the recurrentgemma/jax folder.")
        return

    # Walk through all python files in the jax directory
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                patch_file(filepath)

    print("--- Patching Complete ---")


if __name__ == "__main__":
    main()
