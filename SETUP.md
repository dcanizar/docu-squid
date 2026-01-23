# DocuSquid Setup Guide 🦑

This guide will help you get DocuSquid up and running on your machine in just a few minutes.

## Prerequisites

Before you begin, make sure you have:
- **Python 3.9 or higher** installed ([Download Python](https://www.python.org/downloads/))
- **Git** installed ([Download Git](https://git-scm.com/downloads))
- A terminal/command prompt application

## Step-by-Step Setup

### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/YOUR-USERNAME/DocuSquid.git
cd DocuSquid
```

Replace `YOUR-USERNAME` with your actual GitHub username.

### 2. Create a Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt, indicating the virtual environment is active.

### 3. Install Dependencies

With your virtual environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

This will install all 16 required packages. It may take 2-3 minutes.

### 4. Download NLTK Data (First Time Only)

DocuSquid uses NLTK for text processing. Run this once:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 5. Launch the App

Start the Streamlit web interface:

```bash
streamlit run app.py
```

Your default browser should automatically open to `http://localhost:8501`

If it doesn't open automatically, manually navigate to: **http://localhost:8501**

## Using DocuSquid

### Quick Start Workflow

1. **Upload Documents**
   - Use the sidebar on the left
   - Click "Browse files" or drag-and-drop
   - Supported formats: PDF, DOCX, TXT, MD, HTML, XML, JSON, CSV, RTF, DOC

2. **Set Parameters** (in sidebar)
   - Minimum co-occurrences: How often words must appear together (default: 2)
   - Number of keywords: How many keywords to extract per document (default: 20)

3. **Explore Your Corpus**
   - **Overview Tab**: See statistics about your document collection
   - **Concept Graph Tab**: Visualize connections between keywords
   - **Keywords Tab**: Browse ranked keywords with TF-IDF scores
   - **Documents Tab**: Deep-dive into individual documents
   - **Similarity Tab**: See how documents relate to each other
   - **Search Tab**: Find specific content with regex patterns
   - **Annotations Tab**: Tag and organize keywords with notes

4. **Export Your Results**
   - Download keywords, graph data, or annotations as JSON/CSV
   - Save annotation sessions for later

## Troubleshooting

### Port Already in Use
If you see "Address already in use" error:
```bash
streamlit run app.py --server.port 8502
```

### Module Not Found Error
Make sure your virtual environment is activated (you should see `(venv)` in your prompt):
```bash
# Reactivate if needed
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows
```

### PDF Parsing Issues
If PDFs aren't loading, ensure PyMuPDF installed correctly:
```bash
pip install --upgrade PyMuPDF
```

### DOCX/DOC Issues
For `.doc` files (older Word format), you may need additional tools:
- **macOS/Linux**: Install `pandoc` via your package manager
- **Windows**: Download from [pandoc.org](https://pandoc.org/installing.html)

## Stopping the App

Press `Ctrl + C` in the terminal where Streamlit is running.

To deactivate the virtual environment:
```bash
deactivate
```

## Next Time You Use It

You don't need to repeat the full setup. Just:

1. Navigate to the project folder:
   ```bash
   cd DocuSquid
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Getting Help

- **Issues?** Check the [GitHub Issues](https://github.com/YOUR-USERNAME/DocuSquid/issues)
- **Questions?** Open a new issue with the `question` label
- **Feature Requests?** We'd love to hear them!

## System Requirements

- **Python**: 3.9+
- **RAM**: 4GB minimum (8GB recommended for large document sets)
- **Disk Space**: ~500MB for dependencies
- **Browser**: Chrome, Firefox, Safari, or Edge (modern versions)

---

**Happy analyzing! 🦑📚**
