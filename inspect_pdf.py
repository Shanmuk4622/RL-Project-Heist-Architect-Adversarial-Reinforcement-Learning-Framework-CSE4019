import pdfplumber

with pdfplumber.open(r'Files/sample report.pdf') as pdf:
    for i, page in enumerate(pdf.pages[:4]):   # first 4 pages
        print(f'\n=== PAGE {i+1} ===')
        text = page.extract_text()
        if text:
            print(text[:3000])
