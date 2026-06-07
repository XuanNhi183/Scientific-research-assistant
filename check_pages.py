import fitz, glob

for f in glob.glob("./data/uploads/*.pdf"):
    print(f, fitz.open(f).page_count)
