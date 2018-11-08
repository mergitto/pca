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

def show_result(pca, pca_explain=""):
    for key in pca:
        # 主成分の次元ごとの寄与率を出力する
        print("============   "+key+"  ==============")
        print(pca_explain,':各次元の寄与率{0}'.format(pca[key].explained_variance_ratio_))
        print(pca_explain,':累積寄与率{0}'.format(sum(pca[key].explained_variance_ratio_)))

def to_scatter(transformed, title=""):
    # 主成分をプロットする
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    for key in transformed:
        ax.scatter(transformed[key][:, 0], transformed[key][:, 1], label=key, marker="^") # 適合報告書の主成分分析
    ax.set_title('「'+title+'」の報告書による主成分分析')
    ax.set_xlabel('pc1')
    ax.set_ylabel('pc2')

    ax.legend(loc='upper right')

    # グラフを表示する
    plt.savefig('./images/pca_'+title+'.pdf', format="pdf", dpi=300)
    #plt.show()


def calc_pca(reports_vector=None, input_word=""):
    pca = {}
    transformed = {}
    for key in reports_vector.keys():
        pca[key], transformed[key] = pca_transformed(reports_vector[key])

    show_result(pca, pca_explain="ALL")
    to_scatter(transformed, title=input_word)

if __name__ == '__main__':
    reports_vector = load_pickle('./pickle_data/report_vector_all.pickle')
    calc_pca(reports_vector=reports_vector, input_word="全部")

    type_vector = load_pickle('./pickle_data/type_vector_all.pickle')
    calc_pca(type_vector, input_word="業種")
    shokushu_vector = load_pickle('./pickle_data/shokushu_vector_all.pickle')
    calc_pca(shokushu_vector, input_word="職種")

    cluster_vector = load_pickle('./pickle_data/clusters_vector_all.pickle')
    calc_pca(cluster_vector, input_word="クラスタ")

