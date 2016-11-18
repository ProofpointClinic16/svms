import re
import numpy as np
import matplotlib
from trainTest import create_sets
#execfile("/home/Documents/svms/trainTest/py")
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint


def vectorize(urls):
    clean = []
    for element in urls:
        #print(element)
        clean.append(element["url"].replace('http://', ''))
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(clean)
    return X.toarray()

def parse(filename):
    data = []

    with open(filename) as f:
        for line in f:
            datum = {}

            resultObj = re.search(r"result': u'(.+?)'}", line).group(1)
            urlObj = re.search(r"url': u'(.+?)', ", line).group(1)

            datum['url'] = urlObj
            datum['result'] = resultObj

            data += [datum]
    return data

def feature(data):
    features = []  # Each entry is the feature score for a given sample
    for i in range(len(data)):
        # for entry in data:
        # url = entry["url"]
        url = data[i]["url"]
        periods = url.count('.')

        slashes = url.count('/')
        features.append([float(periods), float(slashes)])

    # pprint(features)
    return features


def target(data):
    target = []
    for i in range(len(data)):
        # TODO: do something with the data that has a result of error
        if data[i]["result"] == 'clean':
            target.append(1.)
        else:
            target.append(0.)
    # pprint(target)
    return target


def format():
    # This will not work without the file. I made a file with the first 500 samples from
    # the big data file.
    #data = parse('big_sample.txt')

    training, testing = create_sets('big_sample.txt')

    feat = vectorize(training + testing)

    half = len(feat) // 2

    trainingFeatures = feat[:half]
    predictionFeatures = feat[half:]
    print('passed features')

    #feat = feature(data)

    tar = target(training + testing)
    trainingTarget = tar[:half]
    predictionTarget = tar[half:]
    print('passed target')
    print(len(feat[0]))
    model = fit(trainingFeatures, trainingTarget)
    print('passed fitting')

    res = []

    for element in predictionFeatures:
        res.append([model.predict([element])])

    print('passed append')
    numErrors = 0

    for i in range(len(res)):
        if predictionTarget[i] != res[i]:
            numErrors += 1
    print(numErrors / len(res) * 100)

    #x = np.array(feat)
    #return fit(x, tar)


def fit(features, target):
    model = svm.SVC(kernel='linear')
    model.fit(features, target)
    #plot(model, features, target)
    return model


# Returns the formatted testing feature matrix
# Then if you have a model and the test data, you can run model.predict(testdata)
# to get predictions for all the data
def testing():
    # I made a file named test-data that contained the next 500 entries in the data file
    # So running this won't work without that file, or a file with the same name
    data = parse('big_sample.txt')
    feat = feature(data)
    x = np.array(feat)
    xScaled = preprocessing.scale(x)
    return xScaled

def plot(model, X, Y):
    # get the separating hyperplane
    w = model.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (model.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = model.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = model.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=80, facecolors='none')
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

    plt.axis('tight')
    plt.show()


# TODO: Clean up code, start making a structure for the code
# TODO: Possibly make some graphics so that we can see what is happening with our data
# TODO: See how our features are doing (using graphics?) and improve upon them
# TODO: Test on a lot of data

if __name__ == '__main__':
    format()
