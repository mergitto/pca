import pickle
from collections import defaultdict
from const import *


def load_advice(load_file_name):
    with open(load_file_name, 'rb') as f:
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
    dictionary['input_word'].append(SHIKAKU)
    dictionary['input_word'].append(SHIKAKU)

def dump_pickle(data, save_file_name):
    with open(save_file_name, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    reports = load_advice("./advice_classification_2.pickle")
    reports_vector = pluck_vector_sum(reports)
    add_vector(reports_vector, key='input_word', value=SHIKAKU)
    dump_pickle(reports_vector, 'report_vector_shikaku.pickle')

    reports = load_advice("./pickle_data/advice_spi_tfidf.pickle")
    reports_vector = pluck_vector_sum(reports)
    add_vector(reports_vector, key='input_word', value=SPI)
    dump_pickle(reports_vector, 'report_vector_spi.pickle')

    reports = load_advice("./pickle_data/advice_mensetsu_tfidf.pickle")
    reports_vector = pluck_vector_sum(reports)
    add_vector(reports_vector, key='input_word', value=MENSETSU)
    dump_pickle(reports_vector, 'report_vector_mensetsu.pickle')


