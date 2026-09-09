import os
import sys

try:
    import pypandoc
except ImportError:
    print("pypandoc not installed. Run: pip install pypandoc")
    sys.exit(1)

input_tex = "main.tex"
output_docx = "main.docx"

if not os.path.exists(input_tex):
    print(f"Error: {input_tex} not found.")
    sys.exit(1)

print(f"Converting {input_tex} to {output_docx} via pandoc...")
try:
    pypandoc.convert_file(input_tex, 'docx', outputfile=output_docx)
    print(f"Conversion complete: {output_docx}")
except Exception as e:
    print(f"Conversion failed or pandoc required: {e}")
    print("Tip: Install pandoc on your system or run: pypandoc.download_pandoc()")
