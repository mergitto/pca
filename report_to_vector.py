import pickle
from collections import defaultdict
from const import *


def load_advice():
    with open("./advice_classification_2.pickle", 'rb') as f:
        return pickle.load(f)

def pluck_vector_sum(reports):
    reports_vector = defaultdict(list)
    for report in reports.values():
        if report['evaluation'] == 'high':
            reports_vector['high'].append(report['vectorSum'])
        else:
            reports_vector['low'].append(report['vectorSum'])
    return reports_vector

def add_vector(dictionary, key=None, value=None):
    return dictionary['input_word'].append(SHIKAKU)

def dump_pickle(data, save_file_name):
    with open(save_file_name, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    reports = load_advice()

    reports_vector = pluck_vector_sum(reports)
    add_vector(reports_vector, key='input_word', value=SHIKAKU)
    add_vector(reports_vector, key='input_word', value=SHIKAKU)

    dump_pickle(reports_vector, 'report_vector_shikaku.pickle')


