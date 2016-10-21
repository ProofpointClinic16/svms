import json
from pprint import pprint


def parse(filename):
    data = []

    with open(filename) as f:
        for line in f:
            line = line.replace("u'","\"")
            line = line.replace("u\"", "\"")
            line = line.replace("':", "\":")
            line = line.replace("',", "\",")
            line = line.replace("']", "\"]")
            line = line.replace("'}", "\"}")
            line = line.replace(" date", " \"date")
            line = line.replace("),", ")\",")
            line = line.replace(")',", ")\",")
            line = line.replace("None", "\"None\"")
            line = line.replace("Object", "\"Object")
            print(line)
            data.append(json.loads(line))

    pprint(data[0]["url"])
    features = []
    target = []
    #for i in range(len(data)):
    for entry in data:
        url = entry["url"]
        url = data[i]["url"]
        periods = url.count('.')
        features.append(periods)
        if data[i]["results"]["result"] == 'clean':
            target.append(1)
        else:
            target.append(0)
    print(features)
    print(target)

if __name__ == '__main__':
    parse('big_sample.txt')

