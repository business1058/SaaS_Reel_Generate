import os
import ast
import sys
from pathlib import Path

def extract_functions_from_file(file_path):
    """Extract all function definitions from a Python file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        print(f"Warning: Could not parse {file_path} due to syntax errors")
        return []
    
    functions = []
    lines = content.split('\n')
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get function start and end lines
            start_line = node.lineno - 1  # ast uses 1-based indexing
            
            # Find the end of the function by looking for the next function or class at the same indentation level
            end_line = len(lines)
            base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
            
            for i in range(start_line + 1, len(lines)):
                line = lines[i]
                if line.strip() == '':
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent and line.strip():
                    end_line = i
                    break
            
            # Extract function code
            function_lines = lines[start_line:end_line]
            function_code = '\n'.join(function_lines)
            
            functions.append({
                'name': node.name,
                'code': function_code,
                'start_line': start_line,
                'end_line': end_line
            })
    
    return functions

def truncate_function(function_code, function_name):
    """Truncate function at halfway point and add payment comment"""
    lines = function_code.split('\n')
    
    # Find the actual function body (skip def line and docstring)
    body_start = 1
    
    # Skip docstring if present
    if len(lines) > 1:
        # Check for triple quotes (simple detection)
        for i in range(1, len(lines)):
            if '"""' in lines[i] or "'''" in lines[i]:
                # Find closing quotes
                quote_type = '"""' if '"""' in lines[i] else "'''"
                if lines[i].count(quote_type) == 2:  # Single line docstring
                    body_start = i + 1
                    break
                else:  # Multi-line docstring
                    for j in range(i + 1, len(lines)):
                        if quote_type in lines[j]:
                            body_start = j + 1
                            break
                break
    
    # Calculate halfway point of the function body
    body_lines = lines[body_start:]
    if len(body_lines) <= 2:
        # If function is too short, just add comment at the end
        truncated_lines = lines + ['    # Full code available after Monday payment confirmation']
    else:
        halfway_point = len(body_lines) // 2
        truncated_lines = lines[:body_start + halfway_point]
        
        # Add payment comment with proper indentation
        indent = '    '  # Default indentation
        if truncated_lines:
            last_line = truncated_lines[-1]
            if last_line.strip():
                indent = ' ' * (len(last_line) - len(last_line.lstrip()))
        
        truncated_lines.append(f'{indent}# Full code available after Monday payment confirmation')
        truncated_lines.append(f'{indent}# Contact for complete implementation')
    
    return '\n'.join(truncated_lines)

def process_python_files(input_folder, output_folder):
    """Process all Python files in the input folder"""
    # Clean paths by removing any trailing quotes or spaces
    input_folder = input_folder.strip().strip('"').strip("'")
    output_folder = output_folder.strip().strip('"').strip("'")
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    print(f"Cleaned input path: {input_path}")
    print(f"Cleaned output path: {output_path}")
    
    # Create output directory if it doesn't exist
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        print(f"Trying to create directory: {output_path}")
        return
    
    # Find all Python files
    python_files = list(input_path.glob('**/*.py'))
    
    if not python_files:
        print(f"No Python files found in {input_folder}")
        return
    
    print(f"Found {len(python_files)} Python files to process...")
    
    for py_file in python_files:
        print(f"Processing: {py_file.name}")
        
        try:
            # Extract functions from the file
            functions = extract_functions_from_file(py_file)
            
            if not functions:
                print(f"  No functions found in {py_file.name}")
                continue
            
            # Create output content
            output_content = []
            output_content.append(f"# Snippet from: {py_file.name}")
            output_content.append(f"# Generated for client preview - Full code available after payment")
            output_content.append(f"# Contact for complete implementation\n")
            
            # Process each function
            for func in functions:
                print(f"  - Truncating function: {func['name']}")
                truncated_code = truncate_function(func['code'], func['name'])
                output_content.append(truncated_code)
                output_content.append('\n')  # Add spacing between functions
            
            # Write to output file
            output_file = output_path / f"snippet_{py_file.name}"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_content))
            
            print(f"  Created: {output_file}")
            
        except Exception as e:
            print(f"Error processing {py_file.name}: {str(e)}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python snippet_generator.py <input_folder> <output_folder>")
        print("Example: python snippet_generator.py ./src ./snippets")
        return
    
    input_folder = sys.argv[1].strip().strip('"').strip("'")
    output_folder = sys.argv[2].strip().strip('"').strip("'")
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        return
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("-" * 50)
    
    process_python_files(input_folder, output_folder)
    print("-" * 50)
    print("Processing complete!")

if __name__ == "__main__":
    main()