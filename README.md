# **Building Applications with Vector Databases**

## **Course Overview**  
This course teaches how to leverage vector databases like **Pinecone** to build six powerful applications. You'll explore how to use **embeddings** for similarity measurement, perform **semantic search**, and enhance **Retrieval-Augmented Generation (RAG)** using **large language models (LLMs)**. The focus is on building AI-driven applications with minimal coding, giving you hands-on experience in vector search techniques.

### **What You’ll Learn**  
- Build six real-world applications using vector databases.  
- Implement **hybrid search** combining text and images.  
- Develop a **facial similarity ranking** system.  

### **Key Applications Covered**  
1. **Semantic Search** – Improve search by retrieving results based on meaning instead of keywords.  
2. **Retrieval-Augmented Generation (RAG)** – Enhance LLM responses using external knowledge sources.  
3. **Recommender System** – Suggest relevant content by combining semantic search and RAG.  
4. **Hybrid Search** – Perform multimodal search using both text and images.  
5. **Facial Similarity** – Rank faces based on feature similarity.  
6. **Anomaly Detection** – Identify unusual patterns in network logs.  

By the end of this course, you'll be equipped to build AI-powered applications using vector databases. 🚀

## **Course Contents**  

### [**1. Semantic Search**]()  
- **Objective:** Build a search tool that retrieves results based on content meaning rather than keywords.  
- **Key Steps:**  
  - Install requirements and import necessary packages.  
  - Setup environment for OpenAI and Pinecone.  
  - Create embeddings and upload them to Pinecone.  
  - Run queries to retrieve relevant results based on semantic similarity.  

### [**2. Retrieval-Augmented Generation (RAG)**]()  
- **Objective:** Integrate Pinecone for efficient document retrieval to enhance LLM responses.  
- **Key Steps:**  
  - Load and preprocess a dataset (e.g., Wikipedia).  
  - Convert documents into embeddings and store them in Pinecone.  
  - Process user queries, retrieve relevant documents, and generate enhanced responses using LLM.  

### [**3. Recommender Systems**]()  
- **Objective:** Build a recommender system using Pinecone to suggest news articles.  
- **Key Steps:**  
  - Create embeddings for article titles and full content.  
  - Store embeddings in Pinecone and perform content-based search.  
  - Retrieve relevant articles based on semantic similarity.  

### [**4. Hybrid Search**]()  
- **Objective:** Implement multimodal search using both text and images for better search accuracy.  
- **Key Steps:**  
  - Create sparse and dense vectors using BM25 (text) and CLIP (images).  
  - Upload the combined embeddings to Pinecone.  
  - Perform hybrid searches using a combination of text and image queries.  

### [**5. Facial Similarity Search**]()  
- **Objective:** Create a system to compare facial features using a database of images.  
- **Key Steps:**  
  - Use **PCA** and **t-SNE** for dimensionality reduction and data visualization.  
  - Generate facial embeddings using DeepFace.  
  - Store embeddings in Pinecone and compare similarity between facial images.  

### [**6. Anomaly Detection**]()  
- **Objective:** Build an anomaly detection system to identify unusual patterns in network logs.  
- **Key Steps:**  
  - Generate sentence embeddings for log entries.  
  - Store embeddings in Pinecone and query for similar logs.  
  - Identify anomalies by comparing logs based on similarity scores.  

## **Notebooks & Resources**  

| Topic                          | Notebook                                |  
|---------------------------------|-----------------------------------------|  
| **Semantic Search**             | [`L1_Semantic_Search.ipynb`]() |  
| **Retrieval-Augmented Generation** | [`L2_Retrieval_Augmented_Generation.ipynb`]() |  
| **Recommender Systems**         | [`L3_Recommender_Systems.ipynb`]() |  
| **Hybrid Search**               | [`L4_Hybrid_Search.ipynb`]() |  
| **Facial Similarity Search**    | [`L5_Facial_Similarity_Search.ipynb`]() |  
| **Anomaly Detection**           | [`L6_Anomaly_Detection.ipynb`]() |  

## **Getting Started**  

### 1. Install required dependencies:  
```bash
pip install pinecone-client openai sentence-transformers
```

### 2. Set up Pinecone:
- Create a Pinecone account and obtain an API key.
- Initialize Pinecone in your Python environment using the provided API key.

### 3. Load and preprocess your dataset:
- Follow the specific steps in each notebook to load datasets and prepare them for use.

### 4. Run the notebooks:
- Execute the notebooks in order, following the instructions for building and testing the applications.

## **References**  
- [Building Applications with Vector Databases Course Link](https://www.deeplearning.ai/short-courses/building-applications-vector-databases/)  

This course provides you with the tools and knowledge to build AI-powered applications using vector search techniques! 🚀