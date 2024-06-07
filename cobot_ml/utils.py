import ast
import os
import shutil


def find_imports(filepath):
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read(), filepath)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    return imports


def copy_files_and_dependencies(main_file, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    main_dir = os.path.dirname(os.path.abspath(main_file))
    files_to_copy = set()
    files_to_copy.add(main_file)

    to_process = [main_file]
    while to_process:
        current_file = to_process.pop()
        imports = find_imports(current_file)
        for module in imports:
            if module.endswith('.py'):
                module_path = os.path.abspath(os.path.join(main_dir, module))
                if os.path.exists(module_path) and module_path not in files_to_copy:
                    files_to_copy.add(module_path)
                    to_process.append(module_path)

    for file_to_copy in files_to_copy:
        shutil.copy(file_to_copy, destination_dir)
