#!/usr/bin/env python3
"""
DocuSquid MVP Phase 1 - Document Loader
Extracts text from PDF, DOCX, TXT files.
"""

import sys
from pathlib import Path
import PyPDF2
from docx import Document
import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import markdown
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
import csv


def load_document(file_path: str) -> str:
    """
    Extract text from supported document types.
    
    Args:
        file_path: Path to PDF, DOCX, or TXT file
        
    Returns:
        Clean extracted text as string
        
    Raises:
        ValueError: Unsupported format or file error
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = file_path.suffix.lower()
    
    try:
        if ext == '.pdf':
            return extract_pdf(file_path)
        elif ext == '.docx':
            return extract_docx(file_path)
        elif ext == '.txt':
            return extract_txt(file_path)
        elif ext == '.md':
            return extract_markdown(file_path)
        elif ext in ['.html', '.htm']:
            return extract_html(file_path)
        elif ext == '.rtf':
            return extract_rtf(file_path)
        elif ext == '.epub':
            return extract_epub(file_path)
        elif ext == '.csv':
            return extract_csv(file_path)
        elif ext in ['.log', '.dat']:
            return extract_plain_text(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}\nSupported: .pdf, .docx, .txt, .md, .html, .htm, .rtf, .epub, .csv, .log, .dat")
    except Exception as e:
        raise ValueError(f"Error extracting {file_path}: {str(e)}")



def extract_pdf(file_path: Path) -> str:
    """Extract text from PDF using PyPDF2."""
    text = []
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:  # Skip empty pages
                text.append(page_text)
    return '\n\n'.join(text)

def extract_docx(file_path: Path) -> str:
    """Extract text from DOCX using python-docx."""
    doc = Document(file_path)
    text = []
    for para in doc.paragraphs:
        if para.text.strip():  # Skip empty paragraphs
            text.append(para.text)
    return '\n\n'.join(text)

def extract_txt(file_path: Path) -> str:
    """Extract text from TXT file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_markdown(file_path: Path) -> str:
    """Extract text from Markdown, converting to plain text."""
    with open(file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    # Convert markdown to HTML, then strip tags
    html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator='\n\n')

def extract_html(file_path: Path) -> str:
    """Extract text from HTML files."""
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(['script', 'style', 'nav', 'footer', 'header']):
        script.decompose()
    
    return soup.get_text(separator='\n\n')

def extract_rtf(file_path: Path) -> str:
    """Extract text from RTF files."""
    with open(file_path, 'r', encoding='utf-8') as f:
        rtf_content = f.read()
    return rtf_to_text(rtf_content)

def extract_epub(file_path: Path) -> str:
    """Extract text from ePub files."""
    import ebooklib
    from ebooklib import epub
    
    book = epub.read_epub(file_path)
    text_parts = []
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Parse HTML content from each chapter
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text_parts.append(soup.get_text(separator='\n\n'))
    
    return '\n\n'.join(text_parts)

def extract_csv(file_path: Path) -> str:
    """Extract text from CSV files (converts to readable text)."""
    import csv
    text_parts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            # Join row cells with spaces
            text_parts.append(' '.join(str(cell) for cell in row if cell.strip()))
    
    return '\n'.join(text_parts)

def extract_plain_text(file_path: Path) -> str:
    """Extract from plain text variants (.log, .dat, etc.)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try latin-1 if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


def preprocess_text(text: str) -> str:
    """
    Clean extracted text for analysis.
    """
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace/newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common noise (optional: customize later)
    text = re.sub(r'[^\w\s\.\!\?]', ' ', text)  # Keep letters, spaces, sentence punctuation
    
    # Trim
    text = text.strip()
    
    return text

def split_sentences(text: str) -> List[str]:
    """
    Split cleaned text into sentences.
    """
    # Simple split on . ! ? with cleanup
    sentences = re.split(r'[.!?]+', text)
    # Filter empty/short sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

def extract_keywords(sentences: List[str], top_n: int = 20) -> dict:
    """
    Extract top keywords using TF-IDF.
    
    Args:
        sentences: List of cleaned sentences
        top_n: Number of keywords to return
        
    Returns:
        dict of {keyword: tfidf_score}
    """
    if len(sentences) == 0:
        return {}
    
    # TF-IDF vectorizer: English stopwords, min doc freq 1
    vectorizer = TfidfVectorizer(
        stop_words='english',
        min_df=1,  # Appear in at least 1 sentence
        lowercase=True,
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'  # Words 3+ letters
    )
    
    # Fit and transform sentences
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Get feature names (all unique words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate mean TF-IDF score per word across sentences
    mean_scores = tfidf_matrix.mean(axis=0).A1
    
    # Create keyword dict sorted by score
    keywords = {}
    for word, score in zip(feature_names, mean_scores):
        keywords[word] = score
    
    # Return top N
    sorted_keywords = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    return sorted_keywords

def build_concept_graph(sentences: List[str], keywords: dict, min_cooccurrence: int = 2):
    """
    Build graph where keywords are nodes, edges = co-occurrence in sentences.
    
    Args:
        sentences: List of cleaned sentences
        keywords: dict of {keyword: score} from extract_keywords
        min_cooccurrence: Minimum times 2 keywords must appear together
        
    Returns:
        NetworkX Graph object
    """
    G = nx.Graph()
    keyword_list = list(keywords.keys())
    
    # Add all keywords as nodes
    for kw in keyword_list:
        G.add_node(kw, score=keywords[kw])
    
    # Count co-occurrences
    cooccur = defaultdict(int)
    for sent in sentences:
        # Find which keywords appear in this sentence
        present = [kw for kw in keyword_list if kw in sent]
        # Connect all pairs
        for i in range(len(present)):
            for j in range(i+1, len(present)):
                pair = tuple(sorted([present[i], present[j]]))
                cooccur[pair] += 1
    
    # Add edges for pairs above threshold
    for (kw1, kw2), count in cooccur.items():
        if count >= min_cooccurrence:
            G.add_edge(kw1, kw2, weight=count)
    
    return G

def plot_graph(G: nx.Graph, output_file: str = "concept_graph.png"):
    """
    Visualize and save the concept graph.
    """
    plt.figure(figsize=(12, 10))
    
    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes (size by TF-IDF score)
    node_sizes = [G.nodes[node].get('score', 0.01) * 5000 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)
    
    # Draw edges (thickness by co-occurrence count)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*0.3 for w in weights], alpha=0.5)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 Graph saved to: {output_file}")
    plt.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 docusquid.py <path/to/document>")
        print("Supported formats:")
        print("  Documents: .pdf, .docx, .rtf, .epub")
        print("  Web: .html, .htm")
        print("  Text: .txt, .md, .log, .dat")
        print("  Data: .csv")
        sys.exit(1)
    
    try:
        file_path = sys.argv[1]
        raw_text = load_document(file_path)
        
        cleaned = preprocess_text(raw_text)
        sentences = split_sentences(cleaned)
        
        print("✅ SUCCESS!")
        print(f"📄 Raw chars: {len(raw_text)} → Cleaned chars: {len(cleaned)}")
        print(f"🔢 Sentences found: {len(sentences)}")
        
        # NEW: Extract keywords
        keywords = extract_keywords(sentences, top_n=15)
        
        print("\n=== TOP KEYWORDS (TF-IDF) ===")
        for i, (kw, score) in enumerate(keywords.items(), 1):
            print(f"{i:2d}. {kw:<15} {score:.4f}")
        
        # NEW: Build concept graph
        graph = build_concept_graph(sentences, keywords, min_cooccurrence=3)
        print(f"\n=== CONCEPT GRAPH ===")
        print(f"Nodes (keywords): {len(graph.nodes())}")
        print(f"Edges (connections): {len(graph.edges())}")
        
        # Plot it
        plot_graph(graph, output_file="concept_graph.png")
        
        print("\n=== FIRST 3 CLEANED SENTENCES ===")
        for i, sent in enumerate(sentences[:3], 1):
            print(f"{i}. {sent}")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
