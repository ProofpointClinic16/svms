import re
import scipy
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from pprint import pprint


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

    #pprint(data)
    return data


def feature(data):
    #pprint(data[0]["url"])
    features = [] # Each entry is the feature score for a given sample
    for i in range(len(data)):
    #for entry in data:
        #url = entry["url"]
        url = data[i]["url"]
        periods = url.count('.')

        slashes = url.count('/')
        features.append([float(periods),float(slashes)])

    #pprint(features)
    return features


def target(data):
    target = []
    for i in range(len(data)):
        # TODO: do something with the data that has a result of error
        if data[i]["result"] == 'clean':
            target.append(1.)
        else:
            target.append(0.)
    #pprint(target)
    return target


def format():
    # This will not work without the file. I made a file with the first 500 samples from
    # the big data file.
    data = parse('half-data.txt')
    feat = feature(data)
    tar = target(data)
    x = np.array(feat)
    xScaled = preprocessing.scale(x)
    pprint(xScaled)
    return fit(xScaled, tar)


def fit(features, target):
    model = svm.SVC()
    model.fit(features, target)
    pprint(model)
    return model

# Returns the formatted feature matrix
# Then if you have a model and the test data, you can run model.predict(testdata)
# to get predictions for all the data
def testing():
    # I made a file named test-data that contained the next 500 entries in the data file
    # So running this won't work without that file, or a file with the same name
    data = parse('test-data.txt')
    feat = feature(data)
    x = np.array(feat)
    xScaled = preprocessing.scale(x)
    return xScaled

# TODO: Clean up code, start making a structure for the code
# TODO: Possibly make some graphics so that we can see what is happening with our data
# TODO: See how our features are doing (using graphics?) and improve upon them
# TODO: Test on a lot of data

if __name__ == '__main__':
    parse('big_sample.txt')

