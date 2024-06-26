import json
from preprocessor import preprocessorPerformer
from collections import defaultdict
import networkx as nx
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from matplotlib import pyplot as plt
import seaborn as sns


def readJsonFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        articles_data = json.load(file)
    return articles_data

def concatenate_article_text(articles_data):
    article_texts = []
    for article in articles_data:
        article_texts.append(article['Article'])
    concatenated_text = "\n\n".join(article_texts)
    return concatenated_text

def save_preprocessed_text(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)


def convert_json_to_preprocessedTxt():
    fashionData = readJsonFile('./dataset/fashion&beauty/scraped_articles.json')
    sportData = readJsonFile('./dataset/Sports/scraped_articles.json')
    educationData = readJsonFile('./dataset/Science_and_education/scraped_articles.json')

    fashtion_txt = concatenate_article_text(fashionData)
    sports_txt = concatenate_article_text(sportData)
    education_txt = concatenate_article_text(educationData)

    preprocess_fashion_txt = preprocessorPerformer(fashtion_txt)
    preprocess_sport_txt = preprocessorPerformer(sports_txt)
    preprocess_education_txt = preprocessorPerformer(education_txt)


    save_preprocessed_text(preprocess_fashion_txt, './dataset/fashion&beauty/preprocessedText.txt')
    save_preprocessed_text(preprocess_sport_txt, './dataset/Sports/preprocessedText.txt')
    save_preprocessed_text(preprocess_education_txt, './dataset/Science_and_education/preprocessedText.txt')


def read_Preprocess_Text(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def divide_into_chunks(text, chunk_size=300):
    chunks = []
    while len(text) > chunk_size:
        chunk, text = text[:chunk_size], text[chunk_size:]
        chunks.append(chunk)
    if text:
        chunks.append(text)
    return chunks

def create_directed_graphs(text):
    word_chunks = divide_into_chunks(text)
    graphs = []
    for chunk in word_chunks:
        graphs.append(create_graph(chunk))
    return graphs

def create_graph(txt):
    G = nx.DiGraph()
    previous_word = None
    for word in word_tokenize(txt):
        if word not in G:
            G.add_node(word)
        if previous_word:
            if G.has_edge(previous_word, word):
                G[previous_word][word]['weight'] += 1
            else:
                G.add_edge(previous_word, word, weight=1)
        previous_word = word
    
    return G

def split_train_test_data(graphs, text, label):
    text_chunks = divide_into_chunks(text)
    
    combined_data = list(zip(graphs, text_chunks, [label] * len(graphs)))
    random.shuffle(combined_data)
    
    num_train = int(0.8 * len(combined_data))
    
    train_data = combined_data[:num_train]
    test_data = combined_data[num_train:]
    
    return train_data, test_data

def find_mcs(graph_list):
    mcs_graph = nx.DiGraph()
    common_nodes = set.intersection(*[set(g.nodes) for g in graph_list])
    mcs_graph.add_nodes_from(common_nodes)
    for node1 in common_nodes:
        for node2 in common_nodes:
            if all(g.has_edge(node1,node2) for g in graph_list):
                mcs_graph.add_edge(node1, node2)
    return mcs_graph


def mcs_distance(graph1, graph2):
    mcs = find_mcs([graph1,graph2])
    return 1 - len(mcs.nodes) / max(len(graph1.edges), len(graph2.edges))

def knn(train_data, test_instance, k):
    distances = []
    for train_instance,text_chunk ,category in train_data:
        distance = mcs_distance(test_instance, train_instance)
        distances.append((category, distance))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    class_counts = defaultdict(int)
    for neighbor in neighbors:
        class_counts[neighbor[0]] += 1
    predicted_class = max(class_counts, key=class_counts.get)
    return predicted_class

def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec


def k_nearest_neighbor(X_train, y_train, X_test, k):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    return y_pred

def plot_confusion_matrix(confMatrix):
  plt.figure(figsize=(8, 6))
  sns.heatmap(confMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=['fashion', 'Sports', 'Science'], yticklabels=['fashion', 'Sports', 'Science'])
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show()


def visualize_graph(G):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=100, edge_color='black', linewidths=1, arrows=True, arrowsize=20)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()



def main():
    fashiontxt = read_Preprocess_Text('./dataset/fashion&beauty/preprocessedText.txt')
    sportstxt = read_Preprocess_Text('./dataset/Sports/preprocessedText.txt')
    sciencetxt = read_Preprocess_Text('./dataset/Science_and_education/preprocessedText.txt')

    fashion_graphs = create_directed_graphs(fashiontxt)
    sports_graphs = create_directed_graphs(sportstxt)
    science_graphs = create_directed_graphs(sciencetxt)

    fashion_train, fashion_test = split_train_test_data(fashion_graphs, fashiontxt ,'fashion')
    sports_train, sports_test = split_train_test_data(sports_graphs, sportstxt ,'sports')
    science_train, science_test = split_train_test_data(science_graphs, sciencetxt ,'science')

    train_data = fashion_train + sports_train + science_train
    test_data = fashion_test + sports_test + science_test

    k_value = 3
    predictions = []
    true_labels = []

    for test_graph,text_chunk,label in test_data:
        predicted_class = knn(train_data, test_graph, k_value)
        predictions.append(predicted_class)
        true_labels.append(label)
        print(f'Predicted Class : {predicted_class}, Actual_Category : {label}')

    print("Classification Report:")
    print(classification_report(true_labels, predictions))

    correct_predictions = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions * 100

    X_train = [entry[1] for entry in train_data]
    y_train = [entry[2] for entry in train_data]
    X_test = [entry[1] for entry in test_data]
    y_test = [entry[2] for entry in test_data]

    X_train_vec, X_test_vec = vectorize_text(X_train, X_test)

    y_pred = k_nearest_neighbor(X_train_vec, y_train, X_test_vec, k_value)
    print(classification_report(y_test, y_pred))
    confMatrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(confMatrix)

    print(f'Accuracy is: {accuracy}')

if __name__ == "__main__":
    main()