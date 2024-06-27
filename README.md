# Graph-Based Text Classification Using KNN

Welcome to the repository for **Graph-Based Text Classification Using KNN**! This project explores an innovative approach to document classification by leveraging graph theory concepts and the K-Nearest Neighbors (KNN) algorithm. By representing documents as directed graphs and extracting features based on common subgraphs, we aim to enhance classification accuracy beyond traditional vector-based models.

## Project Overview

Document classification is a fundamental task in natural language processing (NLP) with numerous real-world applications. In this project, we dive into the intersection of graph theory and machine learning to develop a robust document classification system. Here's a glimpse of what we cover:

- **Data Preprocessing**: We preprocess the text data, including tasks like removing HTML tags, punctuation, numbers, URLs, and stopwords, as well as stemming and lemmatization.

- **Graph Construction**: Each document is transformed into a directed graph, where nodes represent terms and edges denote term relationships based on their sequence in the text.

- **Feature Extraction**: We extract features using frequent subgraph mining techniques to identify common subgraphs within the document graphs.

- **Classification with KNN**: The KNN algorithm is implemented using a distance measure based on the maximal common subgraph (MCS) between document graphs. Test documents are classified based on the majority class of their k-nearest neighbors.

- **Evaluation**: We evaluate the performance of our classification system using metrics such as accuracy, precision, recall, and F1-score, and visualize the results with a confusion matrix.

## Usage

To replicate the experiment or apply the classification system to your own data, follow these steps:

1. **Preprocessing**: Preprocess your text data using the provided preprocessing functions or adapt them to suit your needs.

2. **Graph Construction**: Transform your documents into directed graphs using the `create_graph()` function.

3. **Feature Extraction**: Extract features from the document graphs using frequent subgraph mining techniques.

4. **Classification**: Use the KNN algorithm to classify documents based on their graph representations.

5. **Evaluation**: Evaluate the performance of the classification system using metrics such as accuracy, precision, recall, and F1-score.

## Repository Structure

- `dataset`: Contains preprocessed text data for three categories: fashion & beauty, sports, and science & education.
- `preprocessor.py`: Python module with functions for text preprocessing.
- `graph_classification.py`: Main script for graph-based text classification using KNN.
- `README.md`: You're reading it! Provides an overview of the project and instructions for usage.

## Dependencies

Ensure you have the following dependencies installed to run the code:

- `nltk`
- `textblob`
- `scikit-learn`
- `networkx`
- `matplotlib`
- `seaborn`

## Acknowledgments

This project was inspired by the intersection of graph theory and machine learning and builds upon foundational research in document classification. We extend our gratitude to the open-source community for their invaluable contributions to the libraries and tools used in this project.

**Happy classifying!** 📚🔍
