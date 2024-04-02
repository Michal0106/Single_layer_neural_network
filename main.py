import os
import string

def text_proportion(text):
    text = text.lower()
    alphabet = string.ascii_lowercase
    counts = [text.count(char) for char in alphabet]
    total_counts = sum(counts)
    if total_counts == 0:
        return [0] * len(counts)
    return [count / total_counts for count in counts]

def load_training_data(directory):
    training_data = []
    for language in os.listdir(directory):
        languageDirPath = os.path.join(directory, language)
        for text in os.listdir(languageDirPath):
            textPath = os.path.join(languageDirPath, text)
            with open(textPath, 'r') as file:
                text = file.read()
                proportion = text_proportion(text)
                training_data.append((proportion, language))
    return training_data

def train_perceptrons(training_data):
    perceptrons = {}
    languages = set(label for _, label in training_data)
    for language in languages:
        class_data = [(instance, 1 if label == language else 0) for instance, label in training_data]
        perceptrons[language] = train_perceptron(class_data)
    return perceptrons

def train_perceptron(class_data):
    weights = [0.1] * len(class_data[0][0])
    bias = 0.1
    learning_rate = 0.01
    while True:
        misclassified = 0
        for instance, true_label in class_data:
            predicted = predict(instance, weights, bias)
            error = true_label - predicted
            if error != 0:
                misclassified += 1
                weights = [w + learning_rate * error * x for w, x in zip(weights, instance)]
                bias += learning_rate * error
        if misclassified == 0:
            break
    return weights, bias

def predict(instance, weights, bias):
    activation = sum(x * y for x, y in zip(weights, instance)) + bias
    return 1 if activation >= 0 else 0


def classify_text(text, perceptrons):
    vector = text_proportion(text)
    predictions = {language: predict(vector, weights, bias) for language, (weights, bias) in perceptrons.items()}
    return max(predictions, key=predictions.get)

directory = "/Users/michalmroz/Documents/PJATK/NAI/Single_layer_neural_network/Languages"
training_data = load_training_data(directory)
perceptrons = train_perceptrons(training_data)

test_text = "Das Deutsche ist eine plurizentrische Sprache, enthält also " \
            "mehrere Standardvarietäten in verschiedenen Regionen. Ihr Sprachgebiet" \
            " umfasst Deutschland, Österreich, die Deutschschweiz, Liechtenstein, Luxemburg, " \
            "Ostbelgien, Südtirol, das Elsass und den Nordosten Lothringens sowie Nordschleswig." \
            " Außerdem ist Deutsch eine Minderheitensprache in einigen europäischen und außereuropäischen " \
            "Ländern, z. B. in Rumänien und Südafrika sowie Nationalsprache im afrikanischen Namibia. " \
            "Deutsch ist die meistgesprochene Muttersprache in der Europäischen Union (EU)"

predicted_language = classify_text(test_text, perceptrons)
print("Predicted language:", predicted_language)

load_training_data(directory)