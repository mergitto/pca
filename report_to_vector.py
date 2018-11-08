import pickle
from collections import defaultdict
from const import *

# 全ての報告書の場合は評価データがないので適当に入れるためのモジュール
from random import random


def load_advice(load_file_name):
    with open(load_file_name, 'rb') as f:
        return pickle.load(f)

def pluck_vector_sum(reports):
    reports_vector = defaultdict(list)
    for report in reports.values():
        if 'evaluation' not in report:
            if type(report['vectorSum']) is int: continue
            reports_vector['advice_vector'].append(report['vectorSum'])
            continue

        if report['evaluation'] == 'high':
            reports_vector['high'].append(report['vectorSum'])
        else:
            reports_vector['low'].append(report['vectorSum'])
    return reports_vector

def reports_type_shokushu_vector(reports):
    type_vector = defaultdict(list)
    shokushu_vector = defaultdict(list)
    for report in reports.values():
        if type(report['vectorSum']) is int: continue
        c_type = report['companyType']
        c_shokushu = report['companyShokushu']
        type_vector[c_type].append(report['vectorSum'])
        shokushu_vector[c_shokushu].append(report['vectorSum'])
    return type_vector, shokushu_vector

def reports_cluster_vector(reports):
    clusters_vector = defaultdict(list)
    for report in reports.values():
        if type(report['vectorSum']) is int: continue
        cluster_number = report['cluster']
        clusters_vector[str(cluster_number)].append(report['vectorSum'])
    return clusters_vector

def add_vector(dictionary, key=None, value=None):
    dictionary[key].append(value)
    dictionary[key].append(value)

def dump_pickle(data, save_file_name):
    with open(save_file_name, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    reports = load_advice("./pickle_data/advice_2_tfidf.pickle")
    type_vector, shokushu_vector = reports_type_shokushu_vector(reports)
    reports_vector = pluck_vector_sum(reports)
    dump_pickle(reports_vector, './pickle_data/report_vector_all.pickle')
    dump_pickle(type_vector, './pickle_data/type_vector_all.pickle')
    dump_pickle(shokushu_vector, './pickle_data/shokushu_vector_all.pickle')

    reports_with_cluster = load_advice("./pickle_data/advice_2_cluster.pickle")
    clusters_vector = reports_cluster_vector(reports_with_cluster)
    dump_pickle(clusters_vector, './pickle_data/clusters_vector_all.pickle')

    reports = load_advice("./advice_classification_2.pickle")
    reports_vector = pluck_vector_sum(reports)
    add_vector(reports_vector, key='input_word', value=SHIKAKU)
    dump_pickle(reports_vector, './pickle_data/report_vector_shikaku.pickle')

    reports = load_advice("./pickle_data/advice_spi_tfidf.pickle")
    reports_vector = pluck_vector_sum(reports)
    add_vector(reports_vector, key='input_word', value=SPI)
    dump_pickle(reports_vector, './pickle_data/report_vector_spi.pickle')

    reports = load_advice("./pickle_data/advice_mensetsu_tfidf.pickle")
    reports_vector = pluck_vector_sum(reports)
    add_vector(reports_vector, key='input_word', value=MENSETSU)
    dump_pickle(reports_vector, './pickle_data/report_vector_mensetsu.pickle')

    reports = load_advice("./pickle_data/advice_resume_tfidf.pickle")
    reports_vector = pluck_vector_sum(reports)
    add_vector(reports_vector, key='input_word', value=MENSETSU)
    dump_pickle(reports_vector, './pickle_data/report_vector_resume.pickle')

    reports = load_advice("./pickle_data/advice_communication_tfidf.pickle")
    reports_vector = pluck_vector_sum(reports)
    add_vector(reports_vector, key='input_word', value=MENSETSU)
    dump_pickle(reports_vector, './pickle_data/report_vector_communication.pickle')


