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

def main():
    input_word = reports_vector['input_word'] # 入力のベクトル
    high_report = reports_vector['high'] # 適合報告書のベクトル和
    low_report = reports_vector['low'] # 非適合報告書のベクトル和

    pca, transformed = pca_transformed(input_word)
    pca1, transformed1 = pca_transformed(high_report)
    pca2, transformed2 = pca_transformed(low_report)

    # 主成分をプロットする
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(transformed[:, 0], transformed[:, 1], c="blue", label="input", marker="o") # 入力の主成分分析
    ax.scatter(transformed1[:, 0], transformed1[:, 1], c="red", label="high", marker="^") # 適合報告書の主成分分析
    ax.scatter(transformed2[:, 0], transformed2[:, 1], c="green", label="low", marker="s") # 非適合報告書の主成分分析
    ax.set_title('検索ワードと評価の高低による主成分分析')
    ax.set_xlabel('pc1')
    ax.set_ylabel('pc2')

    ax.legend(loc='upper right')

    # 主成分の次元ごとの寄与率を出力する
    print('High:各次元の寄与率{0}'.format(pca1.explained_variance_ratio_))
    print('High:累積寄与率{0}'.format(sum(pca1.explained_variance_ratio_)))
    print('Low:各次元の寄与率{0}'.format(pca2.explained_variance_ratio_))
    print('Low:累積寄与率{0}'.format(sum(pca2.explained_variance_ratio_)))

    # グラフを表示する
    #plt.savefig('all.pdf', format="pdf", dpi=300)
    plt.show()


if __name__ == '__main__':
    reports_vector = load_pickle('./report_vector.pickle')
    main()
