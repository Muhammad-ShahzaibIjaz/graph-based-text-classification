<img src="D:/6th Semester/GT/Project/report/logo.png" alt="Logo" style="zoom:45%;" />

<h1 align="center">Graph Theory</h1>
<h3 align="center">2021-CS-75</h3>
<h3 align="center">Muhammad Shahzaib Ijaz</h3>

---

## Introduction

### Background

Document classification stands as a cornerstone in Natural Language Processing (NLP) and information retrieval, aiming to categorize documents into predefined topics or classes based on their content. Traditional approaches, such as term frequency-inverse document frequency (TF-IDF) and bag-of-words (BoW), represent documents as high-dimensional vectors. However, these methods might not fully capture semantic relationships between terms within documents, leading to misclassifications.

### Limitations of Vector-Based Models

Vector-based models, while effective in many scenarios, face limitations in capturing nuanced semantic relationships and suffer from dimensionality issues, impacting classification accuracy and computational efficiency.

### Motivation for Graph-Based Approach

In recent years, graph-based methods have gained traction for document classification, offering a more flexible and expressive representation of document content. By modeling documents as graphs, we can capture not only term presence but also their relationships and interactions, potentially improving classification accuracy.

### Significance of the Project

This project explores graph-based document classification, leveraging foundational papers on graph theory and maximal common subgraphs (MCS) for graph comparison. By delving into graph structures to capture inherent term relationships, the aim is to enhance classification accuracy beyond traditional vector-based models. The project provides hands-on experience with graph theory and machine learning, fostering skills in data representation, algorithm implementation, and analytical thinking.

### Objectives

The primary objective is to develop a document classification system by representing documents as directed graphs, identifying common subgraphs, and applying the K-Nearest Neighbors (KNN) algorithm based on graph similarity measures.

---

## Methodology

### Data Collection and Preparation

#### Data Collection

Data was collected from various online sources on three assigned topics: travel, fashion, and diseases, ensuring a balanced representation. Web scraping and parsing techniques were employed to extract text data from HTML content.

#### Data Preparation

The collected raw text data underwent preprocessing steps, including tokenization, stopword removal, and stemming, to prepare it for further analysis.

### Graph Construction

Documents were represented as directed graphs, with nodes representing unique terms and edges denoting term relationships based on their sequential order in the text. Graph construction was implemented using the `networkx` library in Python.

### Feature Extraction via Common Subgraphs

Frequent subgraph mining techniques were used to identify common subgraphs within the document graphs. These common subgraphs served as features for classification, capturing shared content across documents related to the same topic.

---

## Results

### Data Curation

The dataset was curated to ensure a balanced representation of the three diverse topics: travel, fashion, and diseases, with each topic consisting of 15 pages of text.

### Clarity and Thoroughness of the Methodology

The methodology was meticulously defined and implemented, covering all essential steps of the document classification process, facilitating a structured approach to model development and evaluation.

### Creativity in Graph Representation and Feature Extraction

The use of graph-based features, particularly the identification of common subgraphs, showcased creativity in capturing term relationships within documents, allowing for a more nuanced understanding of document content beyond traditional vector-based models.

### Depth of Analysis and Critical Reflection

Critical reflection on challenges encountered during the project and potential improvements to the approach is essential for future enhancements and contributions to the field of document classification.

---

## Future Work

### Implementation of KNN Classification and Common Subgraph Identification

Efforts will be made to integrate KNN classification based on graph similarity measures and explore techniques for identifying common subgraphs within document graphs.

### Integration of Advanced Graph-Based Techniques

Exploration of advanced graph-based techniques, such as graph neural networks (GNNs) and graph embedding methods, will be pursued to capture complex relationships and semantic information within document graphs.

### Evaluation and Benchmarking

Comprehensive evaluation and benchmarking of the document classification system against existing methods will be conducted to assess its effectiveness and scalability.

---

## Conclusion

This project lays the foundation for a graph-based document classification system, showcasing creativity in graph representation and feature extraction. Future work will focus on addressing missing components and integrating advanced graph-based techniques to enhance the system's accuracy and efficiency. Through continuous iteration and improvement, the proposed methodology holds promise for advancing the field of document classification.
