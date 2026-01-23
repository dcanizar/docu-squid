#!/usr/bin/env python3
"""
DocuSquid Phase 2B - Multi-Document Corpus Analysis
Analyze relationships across multiple documents
"""

import streamlit as st
import tempfile
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from collections import defaultdict
import colorsys

# Import our existing functions
from docusquid import (
    load_document, 
    preprocess_text, 
    split_sentences, 
    extract_keywords,
    build_concept_graph
)

st.set_page_config(
    page_title="DocuSquid - Corpus Analysis",
    page_icon="🦑",
    layout="wide"
)

def analyze_document(file_path, filename):
    """Analyze a single document and return structured data."""
    raw_text = load_document(file_path)
    cleaned = preprocess_text(raw_text)
    sentences = split_sentences(cleaned)
    
    return {
        "filename": filename,
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "sentences": sentences,
        "char_count": len(raw_text),
        "sentence_count": len(sentences)
    }

def extract_corpus_keywords(documents, top_n=30):
    """Extract keywords across all documents combined."""
    all_sentences = []
    for doc in documents:
        all_sentences.extend(doc["sentences"])
    
    return extract_keywords(all_sentences, top_n=top_n)

def build_corpus_graph(documents, keywords, min_cooccur=2):
    """Build graph showing keyword relationships across all documents."""
    G = nx.Graph()
    
    # Add keyword nodes
    for kw, score in keywords.items():
        G.add_node(kw, score=score, type='keyword', documents=set())
    
    # Track which documents contain each keyword
    keyword_list = list(keywords.keys())
    
    for doc in documents:
        doc_name = doc["filename"]
        
        # Mark which keywords appear in this doc
        for kw in keyword_list:
            for sent in doc["sentences"]:
                if kw in sent:
                    G.nodes[kw]['documents'].add(doc_name)
                    break
    
    # Build co-occurrence edges within each document
    cooccur = defaultdict(lambda: {"count": 0, "documents": set()})
    
    for doc in documents:
        doc_name = doc["filename"]
        for sent in doc["sentences"]:
            present = [kw for kw in keyword_list if kw in sent]
            for i in range(len(present)):
                for j in range(i+1, len(present)):
                    pair = tuple(sorted([present[i], present[j]]))
                    cooccur[pair]["count"] += 1
                    cooccur[pair]["documents"].add(doc_name)
    
    # Add edges
    for (kw1, kw2), data in cooccur.items():
        if data["count"] >= min_cooccur:
            G.add_edge(kw1, kw2, weight=data["count"], documents=data["documents"])
    
    return G

def get_color_for_doc(index, total):
    """Generate distinct colors for documents."""
    hue = index / total
    rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
    return f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'

def plot_corpus_graph(G, documents):
    """Create fully interactive draggable multi-document graph."""
    if len(G.nodes()) == 0:
        return None
    
    pos = nx.spring_layout(G, k=0.7, iterations=50)
    
    doc_colors = {doc["filename"]: get_color_for_doc(i, len(documents)) 
                  for i, doc in enumerate(documents)}
    
    # Edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        weight = G[edge[0]][edge[1]]['weight']
        docs_with_edge = G[edge[0]][edge[1]]['documents']
        edge_text.append(f"{edge[0]} ↔ {edge[1]}<br>Co-occurs {weight}x<br>In: {', '.join(docs_with_edge)}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='rgba(125,125,125,0.2)'),
        hoverinfo='skip',
        showlegend=False
    )
    
    # Node traces - separate by document for coloring
    node_traces = []
    
    for doc in documents:
        doc_name = doc["filename"]
        color = doc_colors[doc_name]
        
        # Get nodes that appear in this document
        doc_nodes = [node for node in G.nodes() 
                     if doc_name in G.nodes[node].get('documents', set())]
        
        if not doc_nodes:
            continue
        
        node_x = [pos[node][0] for node in doc_nodes]
        node_y = [pos[node][1] for node in doc_nodes]
        
        # Detailed hover info
        node_hover = []
        for node in doc_nodes:
            docs = G.nodes[node]['documents']
            score = G.nodes[node].get('score', 0)
            connections = list(G.neighbors(node))
            hover_text = (
                f"<b>{node}</b><br>"
                f"TF-IDF Score: {score:.4f}<br>"
                f"In documents: {', '.join(sorted(docs))}<br>"
                f"Connected to: {', '.join(connections[:5])}"
                + (f" (+{len(connections)-5} more)" if len(connections) > 5 else "")
            )
            node_hover.append(hover_text)
        
        node_size = [G.nodes[node].get('score', 0.01) * 1000 for node in doc_nodes]
        node_labels = list(doc_nodes)
        
        node_traces.append(
            go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_labels,
                textposition="top center",
                textfont=dict(size=10, color='black'),
                marker=dict(
                    size=node_size,
                    color=color,
                    line=dict(width=2, color='white'),
                    opacity=0.9
                ),
                hovertext=node_hover,
                hoverinfo='text',
                name=doc_name,
                showlegend=True
            )
        )
    
    # Create figure with full interactivity
    fig = go.Figure(data=[edge_trace] + node_traces)
    
    fig.update_layout(
        hovermode='closest',
        margin=dict(b=20,l=20,r=20,t=60),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-1.2, 1.2]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-1.2, 1.2]
        ),
        height=700,
        legend=dict(
            title="Documents",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        # Enable dragmode
        dragmode='pan',
        plot_bgcolor='rgba(245,245,245,0.5)'
    )
    
    # Add instructions
    fig.add_annotation(
        text="💡 Hover over nodes to see connections | Scroll to zoom | Click legend to filter documents",
        xref="paper", yref="paper",
        x=0.5, y=1.05, showarrow=False,
        font=dict(size=11, color="gray")
    )
    
    return fig

def plot_corpus_graph_interactive(G, documents, pos, show_labels=True, size_mult=1.0):
    """Enhanced interactive graph with custom layout."""
    if len(G.nodes()) == 0:
        return None
    
    doc_colors = {doc["filename"]: get_color_for_doc(i, len(documents)) 
                  for i, doc in enumerate(documents)}
    
    # Edges
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(G[edge[0]][edge[1]]['weight'])
    
    # Variable edge width based on weight
    max_weight = max(edge_weights) if edge_weights else 1
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='rgba(150,150,150,0.3)'),
        hoverinfo='skip',
        showlegend=False
    )
    
    # Nodes by document
    node_traces = []
    
    for doc in documents:
        doc_name = doc["filename"]
        color = doc_colors[doc_name]
        
        doc_nodes = [node for node in G.nodes() 
                     if doc_name in G.nodes[node].get('documents', set())]
        
        if not doc_nodes:
            continue
        
        node_x = [pos[node][0] for node in doc_nodes]
        node_y = [pos[node][1] for node in doc_nodes]
        
        # Rich hover text
        node_hover = []
        for node in doc_nodes:
            docs = G.nodes[node]['documents']
            score = G.nodes[node].get('score', 0)
            connections = list(G.neighbors(node))
            
            hover_text = (
                f"<b style='font-size:14px'>{node.upper()}</b><br><br>"
                f"<b>TF-IDF:</b> {score:.4f}<br>"
                f"<b>Documents:</b> {', '.join(sorted(docs))}<br>"
                f"<b>Connections ({len(connections)}):</b><br>"
            )
            
            # List top connections
            for conn in connections[:8]:
                hover_text += f"  • {conn}<br>"
            if len(connections) > 8:
                hover_text += f"  ... +{len(connections)-8} more"
            
            node_hover.append(hover_text)
        
        node_size = [G.nodes[node].get('score', 0.01) * 1000 * size_mult for node in doc_nodes]
        node_labels = list(doc_nodes) if show_labels else [""] * len(doc_nodes)
        
        node_traces.append(
            go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text' if show_labels else 'markers',
                text=node_labels,
                textposition="top center",
                textfont=dict(size=9, color='black', family='Arial Black'),
                marker=dict(
                    size=node_size,
                    color=color,
                    line=dict(width=2, color='white'),
                    opacity=0.85
                ),
                hovertext=node_hover,
                hoverinfo='text',
                name=doc_name[:30],  # Truncate long names
                showlegend=True
            )
        )
    
    fig = go.Figure(data=[edge_trace] + node_traces)
    
    fig.update_layout(
        hovermode='closest',
        margin=dict(b=10,l=10,r=10,t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=750,
        legend=dict(
            title=dict(text="<b>Documents</b>", font=dict(size=12, color='black')),
            yanchor="top", y=0.98,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(color='black')  # ADD THIS LINE
        ),
        dragmode='pan',
        plot_bgcolor='rgba(250,250,250,0.9)'
    )
    
    return fig

def create_pyvis_graph(G, documents, layout_algorithm='spring'):
    """Create fully interactive draggable network with PyVis - nodes stay put."""
    from pyvis.network import Network
    import streamlit.components.v1 as components
    
    # Create network
    net = Network(height='700px', width='100%', bgcolor='#f5f5f5', font_color='black')
    
    # Color mapping
    doc_colors = {doc["filename"]: get_color_for_doc(i, len(documents)) 
                  for i, doc in enumerate(documents)}
    
    # Choose layout algorithm with better spacing
    scale = 800  # Larger scale = more spread out
    
    if layout_algorithm == 'spring':
        pos = nx.spring_layout(G, k=2.5, iterations=100, scale=scale)
    elif layout_algorithm == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, scale=scale)
    elif layout_algorithm == 'circular':
        pos = nx.circular_layout(G, scale=scale)
    elif layout_algorithm == 'shell':
        # Groups by connectivity - highly connected in center
        pos = nx.shell_layout(G, scale=scale)
    elif layout_algorithm == 'spectral':
        pos = nx.spectral_layout(G, scale=scale)
    elif layout_algorithm == 'spiral':
        # Custom spiral layout for maximum separation
        pos = nx.spiral_layout(G, scale=scale, resolution=0.5)
    else:  # random
        pos = nx.random_layout(G)
        # Scale up random positions
        for node in pos:
            pos[node] = pos[node] * scale
    
    # Add nodes with fixed positions
    for node in G.nodes():
        docs = G.nodes[node].get('documents', set())
        score = G.nodes[node].get('score', 0.01)
        
        # Use first document's color
        color = doc_colors[list(docs)[0]] if docs else '#97c2fc'
        
        # Size based on TF-IDF
        size = score * 50 + 10
        
        # Title (hover tooltip) - Clean text version
        connections = list(G.neighbors(node))
        score = G.nodes[node].get('score', 0.01)
        
        # Build clean text tooltip
        title_lines = []
        title_lines.append(f"━━━━━━━━━━━━━━━━━━━━━")
        title_lines.append(f"🔑 {node.upper()}")
        title_lines.append(f"━━━━━━━━━━━━━━━━━━━━━")
        title_lines.append("")
        title_lines.append(f"📊 TF-IDF Score: {score:.4f}")
        title_lines.append("")
        
        # Documents
        title_lines.append(f"📄 Found in {len(docs)} document(s):")
        for doc in sorted(docs):
            display_doc = doc if len(doc) <= 45 else doc[:42] + "..."
            title_lines.append(f"   • {display_doc}")
        title_lines.append("")
        
        # Connections sorted by strength
        if len(connections) > 0:
            connection_weights = {}
            for conn in connections:
                if G.has_edge(node, conn):
                    weight = G[node][conn]['weight']
                    connection_weights[conn] = weight
            
            sorted_connections = sorted(connection_weights.items(), key=lambda x: x[1], reverse=True)
            
            title_lines.append(f"🔗 Top Connections ({len(connections)} total):")
            
            # Show top 15
            for i, (conn, weight) in enumerate(sorted_connections[:15], 1):
                # Strength indicator with clear symbols
                if weight >= 5:
                    strength = "★★★"  # Strong
                elif weight >= 3:
                    strength = "★★☆"  # Medium
                else:
                    strength = "★☆☆"  # Weak
                
                title_lines.append(f"   {i:2d}. {strength} {conn} ({weight}x)")
            
            if len(connections) > 15:
                title_lines.append(f"   ... +{len(connections) - 15} more connections")
        else:
            title_lines.append("🔗 Connections: None")
        
        title = "\n".join(title_lines)

        
        # Get position from layout
        x, y = pos[node]
        
        net.add_node(node, label=node, size=size, color=color, title=title, 
                     x=float(x), y=float(y),
                     font={'size': 12, 'color': 'black', 'face': 'arial'})
    
    # Add edges
    for edge in G.edges():
        weight = G[edge[0]][edge[1]]['weight']
        docs_with_edge = ', '.join(G[edge[0]][edge[1]]['documents'])
        net.add_edge(edge[0], edge[1], value=weight, 
                    title=f"Co-occurs {weight}x in: {docs_with_edge}")
    
    # Physics DISABLED - nodes stay where you drag them
    net.set_options("""
    {
      "physics": {
        "enabled": false
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "hover": true,
        "navigationButtons": true,
        "keyboard": {
          "enabled": true
        }
      },
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {
          "size": 14,
          "face": "arial",
          "bold": {
            "color": "black"
          }
        },
        "shadow": {
          "enabled": true,
          "size": 5,
          "x": 2,
          "y": 2
        }
      },
      "edges": {
        "smooth": {
          "type": "continuous",
          "roundness": 0.5
        },
        "width": 1.5,
        "color": {
          "inherit": false,
          "color": "rgba(150,150,150,0.3)"
        },
        "shadow": {
          "enabled": false
        }
      }
    }
    """)
    
    # Save to HTML
    html_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w')
    net.save_graph(html_file.name)
    
    # Read and return HTML
    with open(html_file.name, 'r') as f:
        html_string = f.read()
    
    Path(html_file.name).unlink()
    
    return html_string

def calculate_document_similarity(documents, keywords):
    """Calculate keyword overlap between documents."""
    doc_keywords = {}
    keyword_list = list(keywords.keys())
    
    for doc in documents:
        doc_kws = set()
        for kw in keyword_list:
            for sent in doc["sentences"]:
                if kw in sent:
                    doc_kws.add(kw)
                    break
        doc_keywords[doc["filename"]] = doc_kws
    
    # Build similarity matrix
    similarity = []
    doc_names = list(doc_keywords.keys())
    
    for i, doc1 in enumerate(doc_names):
        row = []
        for j, doc2 in enumerate(doc_names):
            if i == j:
                row.append(1.0)
            else:
                kw1 = doc_keywords[doc1]
                kw2 = doc_keywords[doc2]
                if len(kw1) == 0 or len(kw2) == 0:
                    row.append(0.0)
                else:
                    jaccard = len(kw1 & kw2) / len(kw1 | kw2)
                    row.append(jaccard)
        similarity.append(row)
    
    return pd.DataFrame(similarity, index=doc_names, columns=doc_names)

def main():
    # Initialize session state for annotations
    if 'keyword_tags' not in st.session_state:
        st.session_state.keyword_tags = {}  # {keyword: {"tag": "Methods", "color": "blue", "note": "..."}}
    
    if 'document_notes' not in st.session_state:
        st.session_state.document_notes = {}  # {filename: "note text"}
    
    if 'tag_categories' not in st.session_state:
        st.session_state.tag_categories = {
            "Core Concept": "#3498db",      # Blue
            "Methodology": "#2ecc71",       # Green
            "Finding/Result": "#f39c12",    # Orange
            "Question/Gap": "#e74c3c",      # Red
            "Important": "#9b59b6",         # Purple
            "Review Later": "#95a5a6",      # Gray
            "Confirmed": "#16a085",         # Teal
            "Uncertain": "#d35400"          # Dark orange
        }
    st.title("🦑 DocuSquid - Corpus Analysis")
    st.markdown("**Multi-Document Concept Extraction & Relationship Mapping**")
    
    # Sidebar settings
    st.sidebar.header("Analysis Settings")
    top_n = st.sidebar.slider("Keywords to extract", 10, 100, 30)
    min_cooccur = st.sidebar.slider("Min co-occurrence", 1, 10, 2)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Supported Formats:**
    - Documents: PDF, DOCX, RTF, ePub
    - Web: HTML, HTM
    - Text: TXT, MD, LOG, DAT
    - Data: CSV
    """)
    
    # Multi-file uploader
    uploaded_files = st.file_uploader(
        "Upload multiple documents to analyze as a corpus",
        type=['pdf', 'docx', 'txt', 'md', 'html', 'htm', 'rtf', 'epub', 'csv', 'log', 'dat'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        documents = []
        
        with st.spinner(f"🔍 Analyzing {len(uploaded_files)} documents..."):
            for uploaded_file in uploaded_files:
                # Save to temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    doc_data = analyze_document(tmp_path, uploaded_file.name)
                    documents.append(doc_data)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            
            # Extract corpus-wide keywords
            keywords = extract_corpus_keywords(documents, top_n=top_n)
            graph = build_corpus_graph(documents, keywords, min_cooccur=min_cooccur)
            similarity = calculate_document_similarity(documents, keywords)
        
        st.success(f"✅ Analyzed {len(documents)} documents across corpus")
        
        # Metrics
        total_chars = sum(doc["char_count"] for doc in documents)
        total_sents = sum(doc["sentence_count"] for doc in documents)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents", len(documents))
        col2.metric("Total Sentences", f"{total_sents:,}")
        col3.metric("Corpus Keywords", len(keywords))
        col4.metric("Connections", len(graph.edges()))
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🕸️ Concept Graph", 
            "📊 Most Relevant Keywords", 
            "📚 Documents", 
            "🔗 Similarity",
            "🔍 Search & Filter",
            "🏷️ Annotations",
            "💾 Export"
        ])

        
        with tab1:
            st.subheader("Cross-Document Concept Network")
            
            # Graph type selector
            graph_type = st.radio(
                "Graph Type",
                ["Interactive (Draggable)", "Plotly (Static Layout)"],
                horizontal=True
            )
            
            if graph_type == "Interactive (Draggable)":
                # Layout algorithm selector
                col_layout, col_info = st.columns([1, 2])
                
                with col_layout:
                    initial_layout = st.selectbox(
                        "Initial Layout",
                        ["spring", "kamada_kawai", "spectral", "circular", "shell", "spiral"],
                        index=1,  # kamada_kawai default
                        help="Choose how nodes are initially positioned"
                    )
                
                with col_info:
                    st.markdown("""
                    **Layout Guide:**
                    - **Kamada-Kawai**: Well-balanced, minimal edge crossing (recommended)
                    - **Spring**: Force-directed, groups related concepts
                    - **Spectral**: Mathematical optimization for clarity
                    - **Circular**: Equal spacing in circle
                    - **Shell**: Concentric rings by connectivity
                    - **Spiral**: Maximum separation
                    """)
                
                st.markdown("""
                **🖱️ Manual Node Positioning:**
                - **Click + drag** any node to reposition it (others stay in place)
                - **Click + drag background** to pan the view
                - **Scroll** to zoom | **Navigation buttons** bottom-right
                - **Hover** over nodes/edges for details
                """)
                
                with st.spinner(f"Generating graph with {initial_layout} layout..."):
                    html_graph = create_pyvis_graph(graph, documents, layout_algorithm=initial_layout)
                    st.components.v1.html(html_graph, height=720, scrolling=False)
            
            else:
                # Plotly (Static Layout) option
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    layout_type = st.selectbox(
                        "Layout algorithm",
                        ["spring", "circular", "kamada_kawai", "random"],
                        index=0
                    )
                with col_b:
                    show_labels = st.checkbox("Show labels", value=True)
                with col_c:
                    node_size_mult = st.slider("Node size", 0.5, 3.0, 1.0, 0.1)
                
                # Rebuild graph with selected layout
                if layout_type == "spring":
                    pos = nx.spring_layout(graph, k=0.7, iterations=50)
                elif layout_type == "circular":
                    pos = nx.circular_layout(graph)
                elif layout_type == "kamada_kawai":
                    pos = nx.kamada_kawai_layout(graph)
                else:
                    pos = nx.random_layout(graph)
                
                fig = plot_corpus_graph_interactive(graph, documents, pos, show_labels, node_size_mult)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToAdd': ['select2d', 'lasso2d']
                    })
                    
                    st.markdown("""
                    **Interaction Guide:**
                    - 🖱️ Hover over keywords to see connections
                    - 🔍 Scroll to zoom | Click + drag to pan
                    - 📌 Click legend to filter documents
                    """)
                else:
                    st.info("No connections found. Try lowering min co-occurrence.")

        
        with tab2:
            st.subheader("Most Relevant Keywords (TF-IDF Ranked)")
            
            # Build enhanced keyword table with metrics
            kw_data = []
            for kw, score in keywords.items():
                docs_with_kw = graph.nodes[kw].get('documents', set())
                
                # Calculate occurrences
                total_occurrences = 0
                max_in_doc = 0
                doc_occurrences = {}
                
                for doc in documents:
                    count = sum(1 for sent in doc['sentences'] if kw in sent)
                    if count > 0:
                        doc_occurrences[doc['filename']] = count
                        total_occurrences += count
                        max_in_doc = max(max_in_doc, count)
                
                avg_per_doc = total_occurrences / len(docs_with_kw) if docs_with_kw else 0
                
                # Connection metrics
                connections = list(graph.neighbors(kw))
                connection_count = len(connections)
                
                # Connection strength (sum of edge weights)
                connection_strength = sum(graph[kw][neighbor]['weight'] for neighbor in connections) if connections else 0
                
                kw_data.append({
                    "Keyword": kw,
                    "TF-IDF Score": round(score, 4),
                    "Total Occurrences": total_occurrences,
                    "Doc Coverage": f"{len(docs_with_kw)}/{len(documents)}",
                    "Coverage %": round(100 * len(docs_with_kw) / len(documents), 1),
                    "Avg per Doc": round(avg_per_doc, 1),
                    "Max in Single Doc": max_in_doc,
                    "Connections": connection_count,
                    "Connection Strength": connection_strength,
                    "Documents": ", ".join(sorted(docs_with_kw))
                })
            
            kw_df = pd.DataFrame(kw_data)
            
            # Add filters
            st.markdown("**Filter Keywords:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_coverage = st.slider(
                    "Min document coverage %", 
                    0, 100, 0, 5,
                    help="Show only keywords appearing in X% of documents"
                )
            with col2:
                min_connections = st.slider(
                    "Min connections", 
                    0, max(kw_df['Connections'].max(), 1), 0,
                    help="Show only keywords with X+ connections"
                )
            with col3:
                sort_by = st.selectbox(
                    "Sort by",
                    ["TF-IDF Score", "Total Occurrences", "Coverage %", "Connections", "Connection Strength"],
                    help="Primary sort column"
                )
            
            # Apply filters
            filtered_df = kw_df[
                (kw_df['Coverage %'] >= min_coverage) &
                (kw_df['Connections'] >= min_connections)
            ].sort_values(by=sort_by, ascending=False)
            
            # Display stats
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Total Keywords", len(kw_df))
            col_b.metric("After Filters", len(filtered_df))
            col_c.metric("Avg Connections", round(kw_df['Connections'].mean(), 1))
            col_d.metric("Avg Coverage", f"{round(kw_df['Coverage %'].mean(), 1)}%")
            
            # Display table with selective columns
            display_columns = [
                "Keyword", 
                "TF-IDF Score", 
                "Total Occurrences", 
                "Doc Coverage",
                "Coverage %",
                "Avg per Doc",
                "Connections",
                "Connection Strength",
                "Documents"
            ]
            
            st.dataframe(
                filtered_df[display_columns], 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "TF-IDF Score": st.column_config.ProgressColumn(
                        "TF-IDF Score",
                        help="Relevance score (higher = more important)",
                        format="%.4f",
                        min_value=0,
                        max_value=kw_df['TF-IDF Score'].max(),
                    ),
                    "Coverage %": st.column_config.ProgressColumn(
                        "Coverage %",
                        help="Percentage of documents containing this keyword",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Connection Strength": st.column_config.NumberColumn(
                        "Connection Strength",
                        help="Sum of co-occurrence weights",
                        format="%d",
                    ),
                }
            )
            
            # Download filtered results
            st.download_button(
                "📥 Download Filtered Keywords (CSV)",
                data=filtered_df.to_csv(index=False),
                file_name="filtered_keywords.csv",
                mime="text/csv"
            )
            
            # Insights section
            with st.expander("📊 Keyword Insights"):
                st.markdown("**Most Connected Keywords (Hub Concepts):**")
                top_connected = kw_df.nlargest(5, 'Connections')[['Keyword', 'Connections', 'Connection Strength']]
                st.dataframe(top_connected, hide_index=True)
                
                st.markdown("**Highest Coverage (Appear in Most Documents):**")
                top_coverage = kw_df.nlargest(5, 'Coverage %')[['Keyword', 'Coverage %', 'Documents']]
                st.dataframe(top_coverage, hide_index=True)
                
                st.markdown("**Most Frequent Overall:**")
                top_freq = kw_df.nlargest(5, 'Total Occurrences')[['Keyword', 'Total Occurrences', 'Avg per Doc']]
                st.dataframe(top_freq, hide_index=True)

        
        with tab3:
            st.subheader("Document Analysis")
            
            # Overall corpus stats
            st.markdown("### Corpus Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            total_words = sum(len(doc['cleaned_text'].split()) for doc in documents)
            avg_words = total_words / len(documents)
            
            col1.metric("Total Documents", len(documents))
            col2.metric("Total Words", f"{total_words:,}")
            col3.metric("Avg Words/Doc", f"{int(avg_words):,}")
            col4.metric("Total Keywords", len(keywords))
            
            st.markdown("---")
            
            # Document comparison table
            st.markdown("### Document Comparison")
            
            doc_comparison = []
            for doc in documents:
                # Find keywords in this document
                doc_keywords = set()
                for kw in keywords.keys():
                    if any(kw in sent for sent in doc['sentences']):
                        doc_keywords.add(kw)
                
                # Calculate unique keywords (only in this doc)
                unique_kws = set()
                for kw in doc_keywords:
                    appears_in = sum(1 for d in documents if any(kw in s for s in d['sentences']))
                    if appears_in == 1:
                        unique_kws.add(kw)
                
                # Word count
                word_count = len(doc['cleaned_text'].split())
                unique_words = len(set(doc['cleaned_text'].split()))
                
                # Keyword density
                kw_density = (len(doc_keywords) / word_count * 100) if word_count > 0 else 0
                
                # Find most similar document
                max_similarity = 0
                most_similar = "None"
                for other_doc in documents:
                    if other_doc['filename'] != doc['filename']:
                        # Use similarity matrix we already calculated
                        sim_value = similarity.loc[doc['filename'], other_doc['filename']]
                        if sim_value > max_similarity:
                            max_similarity = sim_value
                            most_similar = other_doc['filename']
                
                doc_comparison.append({
                    "Document": doc['filename'],
                    "Words": word_count,
                    "Unique Words": unique_words,
                    "Sentences": doc['sentence_count'],
                    "Keywords Found": len(doc_keywords),
                    "Unique Keywords": len(unique_kws),
                    "Keyword Density": round(kw_density, 2),
                    "Most Similar To": most_similar,
                    "Similarity Score": round(max_similarity, 3)
                })
            
            comp_df = pd.DataFrame(doc_comparison)
            
            st.dataframe(
                comp_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Words": st.column_config.NumberColumn(
                        "Words",
                        help="Total word count",
                        format="%d"
                    ),
                    "Keyword Density": st.column_config.NumberColumn(
                        "Keyword Density",
                        help="Keywords per 100 words",
                        format="%.2f%%"
                    ),
                    "Similarity Score": st.column_config.ProgressColumn(
                        "Similarity Score",
                        help="Jaccard similarity with most similar document",
                        format="%.3f",
                        min_value=0,
                        max_value=1
                    )
                }
            )
            
            st.markdown("---")
            
            # Individual document details
            st.markdown("### Individual Document Details")
            
            selected_doc = st.selectbox(
                "Select document to explore",
                [doc['filename'] for doc in documents],
                key="doc_selector"
            )
            
            # Get selected document data
            current_doc = next(doc for doc in documents if doc['filename'] == selected_doc)
            
            # Metrics for selected doc
            col_a, col_b, col_c, col_d = st.columns(4)
            
            word_count = len(current_doc['cleaned_text'].split())
            
            col_a.metric("Characters", f"{current_doc['char_count']:,}")
            col_b.metric("Words", f"{word_count:,}")
            col_c.metric("Sentences", current_doc['sentence_count'])
            col_d.metric("Avg Words/Sentence", round(word_count / max(current_doc['sentence_count'], 1), 1))
            
            # Tabs for different views of the document
            doc_tab1, doc_tab2, doc_tab3, doc_tab4 = st.tabs([
                "📊 Keywords in This Doc",
                "📝 Sample Text",
                "🔗 Connections",
                "📈 Statistics"
            ])
            
            with doc_tab1:
                st.markdown("**Keywords Found in This Document**")
                
                # Get keywords for this document with their scores
                doc_kw_data = []
                for kw, score in keywords.items():
                    count = sum(1 for sent in current_doc['sentences'] if kw in sent)
                    if count > 0:
                        # Check if unique to this doc
                        appears_in_count = sum(1 for d in documents if any(kw in s for s in d['sentences']))
                        is_unique = appears_in_count == 1
                        
                        doc_kw_data.append({
                            "Keyword": kw,
                            "Occurrences": count,
                            "TF-IDF Score": round(score, 4),
                            "Unique to Doc": "✓" if is_unique else "",
                            "Total Docs With": appears_in_count
                        })
                
                doc_kw_df = pd.DataFrame(doc_kw_data).sort_values("Occurrences", ascending=False)
                
                st.dataframe(doc_kw_df, use_container_width=True, hide_index=True)
                
                if len([d for d in doc_kw_data if d["Unique to Doc"] == "✓"]) > 0:
                    st.info(f"✓ = Unique keywords found only in this document")
            
            with doc_tab2:
                st.markdown("**Text Preview**")
                
                preview_length = st.slider("Preview length (characters)", 500, 5000, 2000, 100)
                
                st.text_area(
                    "Cleaned Text",
                    current_doc['cleaned_text'][:preview_length],
                    height=300
                )
                
                st.markdown("**Sample Sentences**")
                num_sentences = st.slider("Number of sentences to show", 3, 20, 10)
                
                for i, sent in enumerate(current_doc['sentences'][:num_sentences], 1):
                    # Highlight keywords in sentence
                    highlighted = sent
                    for kw in keywords.keys():
                        if kw in sent:
                            highlighted = highlighted.replace(kw, f"**{kw}**")
                    st.markdown(f"{i}. {highlighted}")
            
            with doc_tab3:
                st.markdown("**Keywords Connections Within This Document**")
                
                # Build subgraph for just this document's keywords
                doc_keywords = [kw for kw in keywords.keys() 
                               if any(kw in sent for sent in current_doc['sentences'])]
                
                doc_subgraph = graph.subgraph(doc_keywords).copy()
                
                st.metric("Keywords in Document", len(doc_keywords))
                st.metric("Connections Between Them", len(doc_subgraph.edges()))
                
                if len(doc_subgraph.edges()) > 0:
                    # Show top connections
                    edge_list = []
                    for edge in doc_subgraph.edges():
                        weight = doc_subgraph[edge[0]][edge[1]]['weight']
                        edge_list.append({
                            "Keyword 1": edge[0],
                            "Keyword 2": edge[1],
                            "Co-occurrences": weight
                        })
                    
                    edge_df = pd.DataFrame(edge_list).sort_values("Co-occurrences", ascending=False)
                    st.dataframe(edge_df.head(20), use_container_width=True, hide_index=True)
                else:
                    st.info("No keyword co-occurrences found in this document.")
            
            with doc_tab4:
                st.markdown("**Document Statistics**")
                
                # Word frequency distribution
                words = current_doc['cleaned_text'].split()
                word_freq = pd.Series(words).value_counts().head(20)
                
                col_x, col_y = st.columns(2)
                
                with col_x:
                    st.markdown("**Top 20 Most Frequent Words**")
                    freq_df = pd.DataFrame({
                        "Word": word_freq.index,
                        "Count": word_freq.values
                    })
                    st.dataframe(freq_df, hide_index=True)
                
                with col_y:
                    st.markdown("**Vocabulary Stats**")
                    unique_word_count = len(set(words))
                    lexical_diversity = unique_word_count / len(words) if words else 0
                    
                    st.metric("Total Words", len(words))
                    st.metric("Unique Words", unique_word_count)
                    st.metric("Lexical Diversity", f"{lexical_diversity:.2%}")
                    st.caption("Lexical diversity = unique words / total words (higher = more varied vocabulary)")
                    
                    # Sentence length stats
                    sent_lengths = [len(sent.split()) for sent in current_doc['sentences']]
                    avg_sent_length = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
                    
                    st.metric("Avg Sentence Length", f"{avg_sent_length:.1f} words")
                    st.metric("Shortest Sentence", f"{min(sent_lengths) if sent_lengths else 0} words")
                    st.metric("Longest Sentence", f"{max(sent_lengths) if sent_lengths else 0} words")
        
        with tab4:
            st.subheader("Document Similarity Matrix")
            st.markdown("*Jaccard similarity based on shared keywords (0=no overlap, 1=identical)*")
            
            # Format as heatmap
            fig_sim = go.Figure(data=go.Heatmap(
                z=similarity.values,
                x=similarity.columns,
                y=similarity.index,
                colorscale='Blues',
                text=similarity.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            fig_sim.update_layout(height=500)
            st.plotly_chart(fig_sim, use_container_width=True)
        
        with tab5:
            st.subheader("🔍 Search & Filter Corpus")
            
            st.markdown("Search for specific terms, phrases, or patterns across all documents.")
            
            # Search interface
            search_col1, search_col2 = st.columns([3, 1])
            
            with search_col1:
                search_query = st.text_input(
                    "Search query",
                    placeholder="e.g., creatine monohydrate, protein synthesis, etc.",
                    help="Enter words or phrases to find across all documents"
                )
            
            with search_col2:
                search_mode = st.selectbox(
                    "Search mode",
                    ["Contains", "Exact phrase", "Starts with", "Ends with", "Regex"],
                    help="How to match your query"
                )
            
            # Advanced filters
            with st.expander("⚙️ Advanced Filters"):
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                
                with filter_col1:
                    case_sensitive = st.checkbox("Case sensitive", value=False)
                    whole_words = st.checkbox("Whole words only", value=False)
                
                with filter_col2:
                    # Document filter
                    selected_docs = st.multiselect(
                        "Search only in these documents",
                        [doc['filename'] for doc in documents],
                        default=[doc['filename'] for doc in documents],
                        help="Leave all selected to search everywhere"
                    )
                
                with filter_col3:
                    min_sentence_length = st.slider(
                        "Min sentence length (words)",
                        0, 50, 0,
                        help="Filter out short sentences"
                    )
            
            # Perform search
            if search_query:
                st.markdown("---")
                
                # Prepare search
                query = search_query if case_sensitive else search_query.lower()
                
                results = []
                
                for doc in documents:
                    # Skip if document not selected
                    if doc['filename'] not in selected_docs:
                        continue
                    
                    for sent_idx, sentence in enumerate(doc['sentences'], 1):
                        # Apply sentence length filter
                        if len(sentence.split()) < min_sentence_length:
                            continue
                        
                        search_text = sentence if case_sensitive else sentence.lower()
                        match = False
                        
                        # Different search modes
                        if search_mode == "Contains":
                            if whole_words:
                                import re
                                pattern = r'\b' + re.escape(query) + r'\b'
                                match = bool(re.search(pattern, search_text))
                            else:
                                match = query in search_text
                        
                        elif search_mode == "Exact phrase":
                            match = query == search_text
                        
                        elif search_mode == "Starts with":
                            match = search_text.startswith(query)
                        
                        elif search_mode == "Ends with":
                            match = search_text.endswith(query)
                        
                        elif search_mode == "Regex":
                            import re
                            try:
                                flags = 0 if case_sensitive else re.IGNORECASE
                                match = bool(re.search(query, search_text, flags))
                            except re.error as e:
                                st.error(f"Invalid regex pattern: {e}")
                                match = False
                        
                        if match:
                            # Calculate character position
                            char_pos = sum(len(s) + 2 for s in doc['sentences'][:sent_idx-1])
                            
                            # Highlight the query in the sentence
                            highlighted = sentence
                            if search_mode != "Regex":
                                # Simple highlighting
                                if case_sensitive:
                                    highlighted = sentence.replace(search_query, f"**{search_query}**")
                                else:
                                    # Case-insensitive highlighting
                                    import re
                                    pattern = re.compile(re.escape(search_query), re.IGNORECASE)
                                    highlighted = pattern.sub(lambda m: f"**{m.group(0)}**", sentence)
                            
                            results.append({
                                "Document": doc['filename'],
                                "Sentence #": sent_idx,
                                "Char Position": char_pos,  # NEW LINE
                                "Sentence": highlighted,
                                "Length": len(sentence.split()),
                                "Full Sentence": sentence  # For export
                            })

                
                # Display results
                st.markdown(f"### Results: {len(results)} matches found")
                
                if len(results) > 0:
                    # Summary stats
                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                    
                    docs_with_matches = len(set(r['Document'] for r in results))
                    avg_length = sum(r['Length'] for r in results) / len(results)
                    
                    result_col1.metric("Total Matches", len(results))
                    result_col2.metric("Documents", f"{docs_with_matches}/{len(selected_docs)}")
                    result_col3.metric("Avg Sentence Length", f"{avg_length:.1f} words")
                    result_col4.metric("Search Query", f'"{search_query}"')
                    
                    # Group by document
                    st.markdown("#### Matches by Document")
                    
                    doc_groups = {}
                    for result in results:
                        doc_name = result['Document']
                        if doc_name not in doc_groups:
                            doc_groups[doc_name] = []
                        doc_groups[doc_name].append(result)
                    
                    # Sort by number of matches
                    sorted_docs = sorted(doc_groups.items(), key=lambda x: len(x[1]), reverse=True)
                    
                    for doc_name, doc_results in sorted_docs:
                        with st.expander(f"📄 {doc_name} ({len(doc_results)} matches)"):
                            # Get the full document text for position calculation
                            current_doc = next(d for d in documents if d['filename'] == doc_name)
                            full_text = current_doc['cleaned_text']
                            total_chars = len(full_text)
                            
                            for i, result in enumerate(doc_results, 1):
                                # Calculate character position
                                # Find position by reconstructing text up to this sentence
                                char_position = 0
                                for sent_idx in range(result['Sentence #'] - 1):
                                    if sent_idx < len(current_doc['sentences']):
                                        char_position += len(current_doc['sentences'][sent_idx]) + 2  # +2 for spacing
                                
                                # Calculate percentage through document
                                percent_through = (char_position / total_chars * 100) if total_chars > 0 else 0
                                
                                # Estimate page number (assume ~3000 chars per page)
                                est_page = (char_position // 3000) + 1
                                
                                # Get context (sentences before/after)
                                context_sentences = []
                                sent_idx = result['Sentence #'] - 1
                                
                                # Previous sentence
                                if sent_idx > 0:
                                    context_sentences.append(f"*...{current_doc['sentences'][sent_idx - 1]}*")
                                
                                # Current sentence (highlighted)
                                context_sentences.append(f"**→ {result['Sentence']}**")
                                
                                # Next sentence
                                if sent_idx + 1 < len(current_doc['sentences']):
                                    context_sentences.append(f"*{current_doc['sentences'][sent_idx + 1]}...*")
                                
                                # Display match info
                                location_info = f"📍 **Location:** ~Page {est_page} | {percent_through:.1f}% through document | Char {char_position:,} | Sentence #{result['Sentence #']}"
                                
                                st.markdown(f"**Match {i}**")
                                st.caption(location_info)
                                st.markdown(" ".join(context_sentences))
                                st.markdown("---")
                    
                    # Results table view
                    st.markdown("#### All Results (Table View)")
                    
                    results_df = pd.DataFrame([
                        {
                            "Document": r['Document'],
                            "Sentence #": r['Sentence #'],
                            "Preview": r['Full Sentence'][:100] + "..." if len(r['Full Sentence']) > 100 else r['Full Sentence'],
                            "Length": r['Length']
                        }
                        for r in results
                    ])
                    
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Export results
                    st.markdown("#### Export Search Results")
                    
                    export_results = pd.DataFrame([
                        {
                            "Document": r['Document'],
                            "Sentence Number": r['Sentence #'],
                            "Sentence": r['Full Sentence'],
                            "Word Count": r['Length']
                        }
                        for r in results
                    ])
                    
                    col_export1, col_export2 = st.columns(2)
                    
                    with col_export1:
                        st.download_button(
                            "📥 Download Results (CSV)",
                            data=export_results.to_csv(index=False),
                            file_name=f"search_results_{search_query.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                    
                    with col_export2:
                        # JSON export
                        st.download_button(
                            "📥 Download Results (JSON)",
                            data=json.dumps(results, indent=2),
                            file_name=f"search_results_{search_query.replace(' ', '_')}.json",
                            mime="application/json"
                        )
                
                else:
                    st.info(f"No matches found for '{search_query}'. Try adjusting your search mode or filters.")
            
            else:
                st.info("👆 Enter a search query to find sentences across your corpus")
                
                st.markdown(r"""
                **Search Tips:**
                - **Contains**: Finds any occurrence of your term (most flexible)
                - **Exact phrase**: Matches the complete sentence exactly
                - **Starts with**: Finds sentences beginning with your term
                - **Ends with**: Finds sentences ending with your term
                - **Regex**: Advanced pattern matching (e.g., `creatine.*dosage` finds "creatine" followed by "dosage")
                
                **Examples:**
                - Search "protein synthesis" in Contains mode
                - Use regex `\d+\s*(mg|g)` to find all dosage mentions like "20mg", "5 g"
                - Enable "Whole words only" to avoid matching "create" when searching "creatine"
                """)

        with tab6:
            st.subheader("🏷️ Annotations & Tags")
            
            st.markdown("Tag keywords, add notes, and organize your analysis.")
            
            # Two column layout
            anno_tab1, anno_tab2, anno_tab3 = st.tabs([
                "🏷️ Tag Keywords",
                "📝 Document Notes",
                "💾 Save/Load Annotations"
            ])
            
            with anno_tab1:
                st.markdown("### Tag Keywords")
                
                # Tag category manager
                with st.expander("⚙️ Manage Tag Categories"):
                    st.markdown("**Current Tag Categories:**")
                    
                    # Display existing categories
                    for tag_name, tag_color in st.session_state.tag_categories.items():
                        col1, col2, col3 = st.columns([3, 2, 1])
                        col1.markdown(f"**{tag_name}**")
                        col2.color_picker(f"Color for {tag_name}", tag_color, key=f"color_{tag_name}", disabled=True)
                        if col3.button("🗑️", key=f"delete_{tag_name}"):
                            del st.session_state.tag_categories[tag_name]
                            st.rerun()
                    
                    # Add new category
                    st.markdown("**Add New Category:**")
                    new_col1, new_col2, new_col3 = st.columns([3, 2, 1])
                    new_tag_name = new_col1.text_input("Category name", key="new_tag_name")
                    new_tag_color = new_col2.color_picker("Color", "#3498db", key="new_tag_color")
                    if new_col3.button("➕ Add") and new_tag_name:
                        st.session_state.tag_categories[new_tag_name] = new_tag_color
                        st.rerun()
                
                st.markdown("---")
                
                # Keyword tagging interface
                st.markdown("### Tag Keywords")
                
                # Filter keywords
                tag_filter_col1, tag_filter_col2 = st.columns(2)
                
                with tag_filter_col1:
                    filter_by_tag = st.multiselect(
                        "Filter by existing tags",
                        ["Untagged"] + list(st.session_state.tag_categories.keys()),
                        default=[]
                    )
                
                with tag_filter_col2:
                    search_kw = st.text_input("Search keywords", placeholder="Filter keyword list...")
                
                # Build keyword list with tags
                keyword_list = []
                for kw in keywords.keys():
                    # Apply filters
                    if search_kw and search_kw.lower() not in kw.lower():
                        continue
                    
                    current_tag = st.session_state.keyword_tags.get(kw, {}).get("tag", None)
                    
                    if filter_by_tag:
                        if "Untagged" in filter_by_tag and current_tag is not None:
                            continue
                        if current_tag and current_tag not in filter_by_tag:
                            continue
                    
                    keyword_list.append({
                        "keyword": kw,
                        "tag": current_tag,
                        "note": st.session_state.keyword_tags.get(kw, {}).get("note", "")
                    })
                
                st.caption(f"Showing {len(keyword_list)} of {len(keywords)} keywords")
                
                # Display keywords with tagging interface
                for kw_data in keyword_list[:50]:  # Limit to 50 for performance
                    kw = kw_data["keyword"]
                    
                    with st.container():
                        kw_col1, kw_col2, kw_col3, kw_col4 = st.columns([2, 2, 3, 1])
                        
                        # Keyword name with color indicator
                        current_tag = kw_data["tag"]
                        if current_tag:
                            color = st.session_state.tag_categories.get(current_tag, "#cccccc")
                            kw_col1.markdown(f"<span style='background-color:{color}; padding:2px 8px; border-radius:3px; color:white; font-weight:bold;'>{kw}</span>", unsafe_allow_html=True)
                        else:
                            kw_col1.markdown(f"**{kw}**")
                        
                        # Tag selector
                        tag_options = ["(No tag)"] + list(st.session_state.tag_categories.keys())
                        current_index = 0
                        if current_tag:
                            try:
                                current_index = tag_options.index(current_tag)
                            except ValueError:
                                current_index = 0
                        
                        selected_tag = kw_col2.selectbox(
                            "Tag",
                            tag_options,
                            index=current_index,
                            key=f"tag_select_{kw}",
                            label_visibility="collapsed"
                        )
                        
                        # Update tag
                        if selected_tag != "(No tag)":
                            if kw not in st.session_state.keyword_tags:
                                st.session_state.keyword_tags[kw] = {}
                            st.session_state.keyword_tags[kw]["tag"] = selected_tag
                        elif kw in st.session_state.keyword_tags:
                            st.session_state.keyword_tags[kw].pop("tag", None)
                        
                        # Note input
                        note_text = kw_col3.text_input(
                            "Note",
                            value=kw_data["note"],
                            key=f"note_{kw}",
                            placeholder="Add a note...",
                            label_visibility="collapsed"
                        )
                        
                        # Save note
                        if note_text:
                            if kw not in st.session_state.keyword_tags:
                                st.session_state.keyword_tags[kw] = {}
                            st.session_state.keyword_tags[kw]["note"] = note_text
                        
                        # Clear button
                        if kw_col4.button("🗑️", key=f"clear_{kw}"):
                            if kw in st.session_state.keyword_tags:
                                del st.session_state.keyword_tags[kw]
                                st.rerun()
                
                if len(keyword_list) > 50:
                    st.info(f"Showing first 50 keywords. Use filters to narrow down the list.")
                
                # Summary
                st.markdown("---")
                st.markdown("### Tagging Summary")
                
                tag_counts = {}
                for kw_data in st.session_state.keyword_tags.values():
                    tag = kw_data.get("tag", "Untagged")
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                if tag_counts:
                    summary_cols = st.columns(min(len(tag_counts), 4))
                else:
                    summary_cols = st.columns(1)
                    summary_cols[0].info("No tags assigned yet. Tag some keywords above to see statistics.")

                if tag_counts:
                    for i, (tag, count) in enumerate(sorted(tag_counts.items())):
                        summary_cols[i % num_cols].metric(tag, count)
                else:
                    summary_cols[0].info("No tags assigned yet.")

            
            with anno_tab2:
                st.markdown("### Document Notes")
                
                # Select document to annotate
                doc_to_annotate = st.selectbox(
                    "Select document",
                    [doc['filename'] for doc in documents],
                    key="doc_annotate_select"
                )
                
                # Get current note
                current_note = st.session_state.document_notes.get(doc_to_annotate, "")
                
                # Note editor
                st.markdown(f"**Notes for: {doc_to_annotate}**")
                
                new_note = st.text_area(
                    "Document notes",
                    value=current_note,
                    height=200,
                    placeholder="Add observations, questions, or insights about this document...",
                    key=f"doc_note_editor_{doc_to_annotate}"
                )
                
                col1, col2 = st.columns([1, 5])
                
                if col1.button("💾 Save Note"):
                    st.session_state.document_notes[doc_to_annotate] = new_note
                    st.success("Note saved!")
                
                if col2.button("🗑️ Clear Note") and current_note:
                    st.session_state.document_notes[doc_to_annotate] = ""
                    st.rerun()
                
                # Show all document notes
                st.markdown("---")
                st.markdown("### All Document Notes")
                
                if st.session_state.document_notes:
                    for doc_name, note in st.session_state.document_notes.items():
                        if note.strip():
                            with st.expander(f"📄 {doc_name}"):
                                st.markdown(note)
                else:
                    st.info("No document notes yet. Add notes above.")
            
            with anno_tab3:
                st.markdown("### Save & Load Annotations")
                
                st.markdown("""
                Export your annotations to save your work, or import previously saved annotations.
                This includes keyword tags, notes, and document notes.
                """)
                
                # Export annotations
                st.markdown("#### Export Annotations")
                
                annotations_data = {
                    "keyword_tags": st.session_state.keyword_tags,
                    "document_notes": st.session_state.document_notes,
                    "tag_categories": st.session_state.tag_categories,
                    "export_date": pd.Timestamp.now().isoformat(),
                    "documents": [doc['filename'] for doc in documents]
                }
                
                st.download_button(
                    "📥 Download Annotations (JSON)",
                    data=json.dumps(annotations_data, indent=2),
                    file_name="docsquid_annotations.json",
                    mime="application/json"
                )
                
                # Import annotations
                st.markdown("---")
                st.markdown("#### Import Annotations")
                
                uploaded_annotations = st.file_uploader(
                    "Upload previously saved annotations",
                    type=['json'],
                    key="annotation_upload"
                )
                
                if uploaded_annotations:
                    try:
                        imported_data = json.load(uploaded_annotations)
                        
                        st.markdown("**Preview:**")
                        st.json({
                            "Keyword tags": len(imported_data.get("keyword_tags", {})),
                            "Document notes": len(imported_data.get("document_notes", {})),
                            "Tag categories": len(imported_data.get("tag_categories", {})),
                            "Exported": imported_data.get("export_date", "Unknown")
                        })
                        
                        col1, col2 = st.columns(2)
                        
                        if col1.button("✅ Import & Merge", type="primary"):
                            # Merge with existing
                            st.session_state.keyword_tags.update(imported_data.get("keyword_tags", {}))
                            st.session_state.document_notes.update(imported_data.get("document_notes", {}))
                            st.session_state.tag_categories.update(imported_data.get("tag_categories", {}))
                            st.success("Annotations merged successfully!")
                            st.rerun()
                        
                        if col2.button("⚠️ Import & Replace"):
                            # Replace entirely
                            st.session_state.keyword_tags = imported_data.get("keyword_tags", {})
                            st.session_state.document_notes = imported_data.get("document_notes", {})
                            st.session_state.tag_categories = imported_data.get("tag_categories", {})
                            st.success("Annotations replaced successfully!")
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error importing annotations: {e}")
                
                # Clear all annotations
                st.markdown("---")
                st.markdown("#### Clear All Annotations")
                
                if st.button("🗑️ Clear All Annotations", type="secondary"):
                    if st.checkbox("I understand this will delete all tags and notes"):
                        st.session_state.keyword_tags = {}
                        st.session_state.document_notes = {}
                        st.success("All annotations cleared!")
                        st.rerun()

        with tab7:
            st.subheader("Export Corpus Analysis")
            
            export_data = {
                "corpus_stats": {
                    "document_count": len(documents),
                    "total_sentences": total_sents,
                    "keyword_count": len(keywords)
                },
                "documents": [{
                    "filename": doc["filename"],
                    "char_count": doc["char_count"],
                    "sentence_count": doc["sentence_count"]
                } for doc in documents],
                "keywords": keywords,
                "similarity_matrix": similarity.to_dict()
            }
            
            st.download_button(
                "📥 Download Full Analysis (JSON)",
                data=json.dumps(export_data, indent=2, default=str),
                file_name="corpus_analysis.json",
                mime="application/json"
            )
            
            st.download_button(
                "📥 Download Keywords (CSV)",
                data=kw_df.to_csv(index=False),
                file_name="corpus_keywords.csv",
                mime="text/csv"
            )
            
            st.download_button(
                "📥 Download Similarity Matrix (CSV)",
                data=similarity.to_csv(),
                file_name="document_similarity.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👆 Upload 2+ documents to analyze relationships across your corpus")
        st.markdown("""
        **DocuSquid will:**
        - Extract keywords from all documents combined
        - Show which concepts appear in which documents (color-coded graph)
        - Calculate document similarity based on shared concepts
        - Reveal cross-document concept relationships
        """)

if __name__ == "__main__":
    main()
