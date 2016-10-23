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

        # TODO: add another feature so that we have 2 features
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
    fit(xScaled, tar)


def fit(features, target):
    model = svm.SVC()
    model.fit(features, target)
    pprint(model)


if __name__ == '__main__':
    parse('big_sample.txt')

