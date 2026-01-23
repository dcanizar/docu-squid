# DocuSquid 🦑

DocuSquid is a powerful Python application with a Streamlit web interface that analyzes document corpora, extracts keywords and key concepts, and visualizes connections for efficient study and research.

## ✨ Features

### Document Processing
- **Multi-format Support**: PDF, DOC, DOCX, TXT, JSON, XML, MD, HTML, RTF, CSV (10 formats total)
- **Batch Upload**: Drag-and-drop multiple documents for corpus analysis
- **Smart Text Extraction**: Format-specific parsers with preprocessing and cleaning

### Analysis & Visualization
- **Interactive Concept Graphs**: 
  - PyVis draggable network with 6 layout algorithms (Kamada-Kawai, Spring, Spectral, etc.)
  - Plotly static graphs with zoom, pan, and legend filtering
  - Color-coded by document with relationship strength visualization
- **TF-IDF Keyword Extraction**: Ranked keywords with frequency metrics and filtering
- **Document Similarity Analysis**: Heatmap showing content overlap between documents
- **Cross-Document Comparison**: Side-by-side statistics and shared concepts

### Advanced Features
- **Smart Search & Filter**: 
  - Regex-powered search across all documents
  - Context snippets with match highlighting
  - Document and location filtering
- **Annotations & Tagging**: 
  - Tag keywords with custom labels
  - Add notes and observations
  - Save/load annotation sessions (JSON)
- **Export Options**: Download keywords, graph data, and annotations as JSON/CSV

### Web Interface
7 interactive tabs powered by Streamlit:
1. **📊 Overview** - Corpus statistics and metrics
2. **🕸️ Concept Graph** - Interactive network visualization
3. **🔑 Keywords** - TF-IDF ranked terms with filters
4. **📚 Documents** - Individual analysis and comparison
5. **🔗 Similarity** - Document relationship heatmap
6. **🔍 Search** - Advanced regex search with context
7. **🏷️ Annotations** - Tag and annotate keywords

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/DocuSquid.git
   cd DocuSquid
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

**Streamlit Web Interface (Recommended):**
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`

**Command Line (Legacy):**
```bash
python main.py --corpus ./documents/
```

## 📋 Key Dependencies

- **Streamlit** - Web interface
- **NetworkX** - Graph analysis
- **PyVis** - Interactive network visualization
- **Plotly** - Static graph rendering
- **scikit-learn** - TF-IDF and NLP processing
- **PyMuPDF** - PDF parsing
- **python-docx** - DOCX handling
- **BeautifulSoup4** - HTML/XML parsing
- **pandas** - Data manipulation

## 🎯 Usage

1. **Upload Documents**: Use the sidebar to upload multiple files (10 format types supported)
2. **Explore Overview**: View corpus statistics and document summaries
3. **Visualize Concepts**: Generate interactive or static concept graphs
4. **Analyze Keywords**: Filter and sort by TF-IDF scores
5. **Compare Documents**: View similarity heatmap and shared concepts
6. **Search Content**: Use regex patterns to find specific content
7. **Annotate & Tag**: Add notes and organize keywords with custom tags
8. **Export Data**: Download results in JSON or CSV format

## 🛠️ Project Structure

```
DocuSquid/
├── app.py                 # Streamlit web interface
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
├── src/
│   ├── parsers/          # Format-specific document parsers
│   ├── extractor/        # Keyword and concept extraction
│   ├── preprocessor/     # Text cleaning and normalization
│   └── visualizer/       # Graph generation utilities
├── documents/            # Sample corpus (optional)
└── output/               # Generated results
```

## 🗺️ Development Roadmap

### Completed ✅
- [x] Multi-format document parsing (10 types)
- [x] TF-IDF keyword extraction
- [x] Interactive web UI with Streamlit
- [x] Dual graph visualization (PyVis + Plotly)
- [x] Document similarity analysis
- [x] Advanced search with regex
- [x] Annotations and tagging system
- [x] Export functionality (JSON/CSV)

### Planned Features 🔮
- [ ] LLM integration for semantic analysis
- [ ] Custom stopwords upload
- [ ] Boolean search (AND/OR/NOT operators)
- [ ] Keyword frequency heatmap
- [ ] Citation extraction and linking
- [ ] PDF report export with embedded graphs
- [ ] Performance optimization (caching, batch processing)

## 🤝 Contributing

Contributions welcome! Areas of focus:
- New document format parsers
- LLM prompt engineering for concept extraction
- Visualization enhancements
- Performance optimizations
- UI/UX improvements

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with Python, Streamlit, NetworkX, and the open-source community.

---

**Made with 🦑 for researchers, students, and knowledge workers**
