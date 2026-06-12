import os
from pathlib import Path


def generate_tree(dir_path: Path, prefix: str = "", ignore_list=None):
    """
    Recursively generates a visual tree structure of a directory.
    """
    if ignore_list is None:
        # Standard folders to ignore to keep the text file clean
        ignore_list = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            ".env",
            "dist",
            "build",
            ".DS_Store",
        }

    try:
        # Sort so directories appear before files, alphabetically
        items = sorted(
            list(dir_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower())
        )
    except PermissionError:
        return []

    # Filter out ignored items
    items = [item for item in items if item.name not in ignore_list]

    tree_lines = []
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "

        # Add the current file/folder
        tree_lines.append(
            f"{prefix}{connector}{item.name}/"
            if item.is_dir()
            else f"{prefix}{connector}{item.name}"
        )

        # If it's a directory, recurse into it
        if item.is_dir():
            next_prefix = prefix + ("    " if is_last else "│   ")
            tree_lines.extend(generate_tree(item, next_prefix, ignore_list))

    return tree_lines


def save_directory_txt(output_file="directory.txt"):
    """
    Gets the current working directory and writes its structure to a file.
    """
    root_dir = Path.cwd()

    # Header for the text file
    tree = [f"Root: {root_dir.name}/", "═" * 40]
    tree.extend(generate_tree(root_dir))

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tree))

    print(f"✔ Clean directory tree successfully written to: {root_dir / output_file}")


if __name__ == "__main__":
    save_directory_txt()
