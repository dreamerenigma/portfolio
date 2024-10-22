import random
import json
import torch

# Initialize a cache dictionary to store responses
cache = {}

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents-en.json', 'r', encoding='utf-8') as json_data:
    intents_en = json.load(json_data)

with open('intents-ru.json', 'r', encoding='utf-8') as json_data:
    intents_ru = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    # Проверяем, есть ли ответ в кэше
    if msg in cache:
        return cache[msg]

    # Токенизируем сообщение
    sentence = tokenize(msg)
    # Преобразуем в Bag of Words
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Получаем выход модели
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    # Получаем тег
    tag = tags[predicted.item()]

    # Получаем вероятности
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        # Проверяем тег в английском файле intents_en
        for intent in intents_en['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                # Кэшируем ответ
                cache[msg] = response
                return response
        # Проверяем тег в русском файле intents_ru
        for intent in intents_ru['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                # Кэшируем ответ
                cache[msg] = response
                return response

    response = "Я не понимаю..."
    # Кэшируем ответ
    cache[msg] = response
    return response

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
