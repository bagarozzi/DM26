# Project Overview: Hiking Trail Recommender (Module 3 Track)
To comply with the "Data Mining Project" guidelines, this proposal outlines a small project in colab or jupyter focusing on topics from the text mining and nlp module. The final deliverable will require a written relation of at least 8 pages, which will be discussed during the oral exam. As mandated, this specific elaborate must be agreed with the teacher prior to execution.

## Core Methodological Approaches
Since the project falls under Module 3, it must utilize unstructured text (e.g., trail reviews, descriptions) and advanced NLP techniques rather than relying solely on tabular data mining. Here are the primary architectural choices available for your proposal:

### Approach A: Retrieval-Augmented Generation (RAG) with Dense Retrieval

This approach converts structured trail data and unstructured reviews into highly searchable text chunks.
- Data Verbalization: Structured data (length, elevation, difficulty) is converted into canonical natural language text chunks (e.g., "This trail is 10km long...") and combined with text from hiker reviews.
- Embedding & Indexing: The text is mapped into Word embeddings and stored in a vector database.
- Retrieval: When a user queries the system, it performs Retrieval-augmented Generation with VectorDB and LLM components to find semantically similar trails and generate a response.   
- Technologies: Hugging Face framework (for embedding models), Milvus (for the VectorDB), and LangChain or LlamaIndex (for pipeline orchestration).   

### Approach B: LLM Agent with Hybrid Search and Function Calling
This approach keeps structured data in a strict database format while using text solely for semantic context.
- Query Parsing: An LLM interprets the user's natural language request to separate hard constraints from semantic preferences.
- Execution: The system leverages Function Call Generation to execute hard filters on structured metadata (e.g., length < 15km or state == 'Alaska') while simultaneously performing semantic search on the text reviews.
- Technologies: Generative Large Language Model, LLM Agents, Prompt Engineering, and LangChain/LlamaIndex.

### Approach C: Graph Machine Learning (Advanced)
This approach maps the relationships between hikers, trails, geographical features, and activities.
- Architecture: Trails, parks, and tags become interconnected nodes, while user preferences and past hikes form edges.
- Technologies: Graph Machine Learning (GNN, Graph Embedding and GraphRAG).   

## Step-by-Step Implementation Plan
### 1. Data Acquisition & Preparation
- Scrape or acquire textual data, prioritizing trail descriptions and user reviews.
- Apply Text Representation, Sentiment Analysis and Opinion Mining to extract qualitative insights from hiker reviews, such as identifying commonly mentioned hazards or scenic highlights.   
### 2. Knowledge Base Construction 
- Format the combined data using canonical text templates to blend numeric statistics with review text.
- Generate Word embeddings using open source tools like the Hugging Face framework.
- Store the resulting embeddings and metadata filters in Milvus.
### 3. Chatbot / Recommender Development
- Develop the system entirely within a Colab or Jupyter notebook environment.
- Implement the core logic using LangChain or LlamaIndex.
- Apply Prompt Engineering to guide the Generative Large Language Model's persona, ensuring it explains why a trail matches the user's physical stats. 
### 4. System Evaluation
- Implement rigorous Evaluation Methods to measure the system's accuracy and reliability.
- Suggested Framework: Use RAG-specific evaluation metrics (such as the "RAG Triad" of Context Relevance, Groundedness, and Answer Relevance via tools like RAGAS or TruLens) to prove the LLM does not hallucinate non-existent trail features.