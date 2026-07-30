import pypandoc

print("Downloading pandoc...")
pypandoc.download_pandoc()

print("Converting LaTeX to Word...")
pypandoc.convert_file('wildfire_paper_final.tex', 'docx', outputfile='wildfire_paper_final.docx')
print("Conversion complete.")
