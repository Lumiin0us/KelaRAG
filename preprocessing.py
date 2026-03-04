import pymupdf # imports the pymupdf library
import re
import os
import json

dataset = "./dataset/"
dataset = os.listdir(dataset)

#util methods 
def clearWhiteSpace(text):
  return " ".join(text.split())

def clearPageNumbers(text):
    pattern = r"\d+\s+\d+$"
    clean_text = re.sub(pattern, "", text)
    clean_text = clean_text.replace("  ", " ").strip()
    return clean_text

def clearJunkPages(text):
    t = text.strip().lower()
    if not t:
       return True

    if len(t) < 200: 
       return True 
    
    if "table of contents" in t or re.search(r"\bcontents\b", t):
        return True
    
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    dotted = sum(1 for ln in lines if "...." in ln or "·" in ln)
    if dotted / len(lines) > 0.25:
        return True
    
    return False
  

for data in dataset:
    pdfData = {}
    doc = pymupdf.open(f"./dataset/{data}")
    for page in range(0, len(doc)): 
        text = doc[page].get_text() 
        text = clearWhiteSpace(text)
        if clearJunkPages(text) == False:
            text = clearPageNumbers(text)
            pdfData[data + "_" + str(page)] = {"source": data, "content": text, "pageNumber": page + 1}
    
    os.makedirs("preprocessedDataset", exist_ok=True)

    with open(f"preprocessedDataset/{data}.json", "w", encoding="utf-8") as f:
        json.dump(pdfData, f, ensure_ascii=False, indent=2)


#format [source: "", text: "", pageNumber: 0]


# echo "# KelaRAG" >> README.md
# git init
# git add README.md
# git commit -m "first commit"
# git branch -M main
# git remote add origin https://github.com/Lumiin0us/KelaRAG.git
# git push -u origin main