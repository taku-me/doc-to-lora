#!/usr/bin/env python3
import os
import glob

# Find the PDF file
pdf_dir = '/Users/taku.me/Library/CloudStorage/Dropbox/_inbox/Files/Doc/book/'
pdf_files = glob.glob(os.path.join(pdf_dir, 'xTalk*.pdf'))
print(f"Found PDF files: {pdf_files}")

if not pdf_files:
    print("No PDF files found!")
    exit(1)

# Use the first file (not the (1) copy)
pdf_path = None
for f in pdf_files:
    if '(1)' not in f:
        pdf_path = f
        break
if not pdf_path:
    pdf_path = pdf_files[0]

print(f"Using: {pdf_path}")

import PyPDF2
with open(pdf_path, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    num_pages = len(reader.pages)
    print(f'Pages: {num_pages}')
    
    # Extract all text
    all_text = []
    for i in range(num_pages):
        text = reader.pages[i].extract_text()
        all_text.append(text)
        if i < 3:
            print(f'--- Page {i+1} ---')
            print(text[:500])
            print()
    
    # Save to file
    output_path = '/Volumes/NVME202502/projects/doc-to-lora/tmp_xtalk_text.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_text))
    print(f'Saved to {output_path}')
    print(f'Total characters: {sum(len(t) for t in all_text)}')
