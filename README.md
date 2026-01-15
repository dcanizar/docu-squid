# DocuSquid README

DocuSquid is a Python app that analyzes document corpora, extracts keywords and key concepts, and visualizes connections for efficient study and research. It supports multiple formats including PDF, DOC, DOCX, TXT, JSON, XML, and MD files, with LLM-powered insights.

## Features
- **Multi-format Support**: Handles PDF, DOC, DOCX, TXT, JSON, XML, MD files seamlessly.
- **Keyword & Concept Extraction**: Identifies important terms and themes from documents.
- **LLM Analysis**: Leverages large language models for semantic understanding and summarization.
- **Concept Mapping**: Connects related ideas across documents for easy visualization and study.
- **Extensible Design**: Modular structure for adding new analysis features.

## Quick Start
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/docu-squid.git
   cd docu-squid
   ```
2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   python main.py --corpus ./documents/
   ```

## Installation
Detailed setup including optional LLM providers (e.g., OpenAI, local models via Ollama).

- Ensure Python 3.9+ is installed.
- Key libraries: `PyMuPDF` (PDFs), `python-docx` (DOCX), `openai` or `langchain` (LLM).

## Usage
```
python main.py --input-dir ./docs --output-dir ./output --llm-provider openai
```
- `--input-dir`: Folder with documents to analyze.
- `--output-dir`: Where to save extractions and maps.
- `--llm-provider`: Choose LLM backend.

## Project Structure
```
docu-squid/
├── src/
│   ├── parsers/     # Format-specific parsers
│   ├── extractor/   # Keyword/concept logic
│   ├── llm/         # LLM integration
│   └── visualizer/  # Concept graphs
├── documents/       # Sample corpus
├── output/          # Generated results
├── requirements.txt
└── main.py
```

## Development Roadmap
- [ ] Basic parsing for all formats.
- [ ] LLM integration for concept extraction.
- [ ] Interactive web UI for exploration.
- [ ] Export to mindmaps (e.g., Mermaid diagrams).

## Contributing
Fork the repo, create a feature branch, and submit a pull request. Focus areas: new parsers, LLM prompts, visualization tools.

## License
MIT License – see [LICENSE](LICENSE) file.[](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)

Copy the content above into your repository's README.md file via GitHub's web editor or locally with `git add README.md`, `git commit -m "Add initial README"`, and `git push`. This structure follows Python project best practices and highlights your core features.[](https://www.makeareadme.com/)
