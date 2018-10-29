#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pickle

def load_pickle(import_file_name):
    with open(import_file_name, 'rb') as f:
        return pickle.load(f)

def pca_transformed(vector):
     # 主成分分析する
     pca = PCA(n_components=2)
     pca.fit(vector)
     # 分析結果を元にデータセットを主成分に変換する
     transformed = pca.fit_transform(vector)
     return pca, transformed

def show_result(pca_1, pca_2, pca_1_explain="", pca_2_explain=""):
    # 主成分の次元ごとの寄与率を出力する
    print(pca_1_explain,':各次元の寄与率{0}'.format(pca_1.explained_variance_ratio_))
    print(pca_1_explain,':累積寄与率{0}'.format(sum(pca_1.explained_variance_ratio_)))
    print(pca_2_explain,':各次元の寄与率{0}'.format(pca_2.explained_variance_ratio_))
    print(pca_2_explain,':累積寄与率{0}'.format(sum(pca_2.explained_variance_ratio_)))

def to_scatter(transformed, transformed_1, transformed_2, title=""):
    # 主成分をプロットする
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(transformed[:, 0], transformed[:, 1], c="blue", label="input", marker="o") # 入力の主成分分析
    ax.scatter(transformed_1[:, 0], transformed_1[:, 1], c="red", label="high", marker="^") # 適合報告書の主成分分析
    ax.scatter(transformed_2[:, 0], transformed_2[:, 1], c="green", label="low", marker="s") # 非適合報告書の主成分分析
    ax.set_title('「'+title+'」と適合・非適合報告書による主成分分析')
    ax.set_xlabel('pc1')
    ax.set_ylabel('pc2')

    ax.legend(loc='upper right')

    # グラフを表示する
    plt.savefig('./images/pca_'+title+'.pdf', format="pdf", dpi=300)
    plt.show()


def calc_pca(reports_vector=None, input_word=""):
    pca, transformed = pca_transformed(reports_vector['input_word'])
    pca1, transformed1 = pca_transformed(reports_vector['high'])
    pca2, transformed2 = pca_transformed(reports_vector['low'])

    show_result(pca1, pca2, pca_1_explain="High", pca_2_explain="Low")
    to_scatter(transformed, transformed1, transformed2, title=input_word)

if __name__ == '__main__':
    reports_vector = load_pickle('./pickle_data/report_vector_shikaku.pickle')
    calc_pca(reports_vector=reports_vector, input_word="資格")

    reports_vector = load_pickle('./pickle_data/report_vector_spi.pickle')
    calc_pca(reports_vector=reports_vector, input_word="spi")

    reports_vector = load_pickle('./pickle_data/report_vector_mensetsu.pickle')
    calc_pca(reports_vector=reports_vector, input_word="面接")

    reports_vector = load_pickle('./pickle_data/report_vector_communication.pickle')
    calc_pca(reports_vector=reports_vector, input_word="コミュニケーション")

    reports_vector = load_pickle('./pickle_data/report_vector_resume.pickle')
    calc_pca(reports_vector=reports_vector, input_word="履歴書")
