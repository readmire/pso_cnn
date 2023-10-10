from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

# train_path = 'input_figshare/train/'
# test_path = 'input_figshare/test'
#
# train_path = 'input_bt_46/train/'
# test_path = 'input_bt_46/test'

train_path = 'input_bt_merged/train/'
test_path = 'input_bt_merged/test'
# batch size, epoch and image input size defined
bs, epo, ims = 16, 20, 224

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(ims, ims),
                                            batch_size=bs,
                                            class_mode='categorical',
                                            shuffle=False)
labels = test_set.classes

# modelNames = os.listdir('bt_art_models/bt_figshare/')
# modelNames = os.listdir('bt_art_models/bt_46/')

modelNames = os.listdir('bt_art_models/bt_merged/')
modelPredictions = list()
print("Model Names: ", modelNames)

for name in modelNames:
    model = load_model(('bt_art_models/bt_merged/' + name))
    modelPrediction = model.predict(test_set)
    modelPredictions.append(modelPrediction)


def generatePopulation(populationNumber):
    population = list()
    for i in range(populationNumber):
        weights = list()
        for j in range(len(modelNames)):
            weights.append(random.random())

        sumOfWeights = sum(weights)

        for i in range(len(weights)):
            weights[i] /= sumOfWeights
        population.append(list(weights))
    return population


def calculatePopulationScore(population):
    results = list()
    for person in population:
        newPredictions = person[0] * modelPredictions[0]
        for i in range(1, len(modelNames)):
            newPredictions += person[i] * modelPredictions[i]
        newPredictions = np.argmax(newPredictions, axis=-1)
        loss = log_loss(test_set.classes, newPredictions)
        results.append(loss)
    return results


def main():
    velocityWeight = 0.7
    c1 = 1.5
    c2 = 1.5
    lowerLimit = 0
    upperLimit = 1
    velocityLimit = (upperLimit - lowerLimit) / 2

    population = generatePopulation(100)
    results = calculatePopulationScore(population)

    personalBestValues = population
    personalBestResults = results

    globalBestResult = max(results)
    idx = results.index(globalBestResult)
    globalBestValue = population[idx]

    plotValues = list()
    plotValues.append(globalBestResult)

    velocity = np.zeros([len(population), len(modelNames)])

    for i in range(1000):
        for j in range(len(population)):
            for k in range(len(modelNames)):
                x = (velocityWeight * velocity[j][k]) + (
                            c1 * random.random() * (population[j][k] - personalBestValues[j][k])) + (
                                c2 * random.random() * (population[j][k] - globalBestValue[k]))
                if x > velocityLimit:
                    x = velocityLimit
                elif x < -velocityLimit:
                    x = -velocityLimit
                velocity[j][k] = x

        population += velocity

        for j in range(len(population)):
            sumOfWeights = sum(population[j])
            for k in range(len(modelNames)):
                population[j][k] /= sumOfWeights

        for j in range(len(population)):
            for k in range(len(modelNames)):
                if population[j][k] > upperLimit:
                    population[j][k] = upperLimit
                elif population[j][k] < lowerLimit:
                    population[j][k] = lowerLimit

        results = calculatePopulationScore(population)

        for j in range(len(population)):
            if results[j] < personalBestResults[j]:
                personalBestResults[j] = results[j]
                personalBestValues[j] = population[j]

        if min(results) <= globalBestResult:
            globalBestResult = min(results)
            idx = results.index(globalBestResult)
            globalBestValue = population[idx]

        plotValues.append(globalBestResult)

    plt.plot(plotValues)
    plt.xlabel("iteration")
    plt.show()
    print(globalBestValue)
    print(globalBestResult)


main()











