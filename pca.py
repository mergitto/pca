#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pickle

def loadVectorSum(pathName):
    with open(pathName, 'rb') as f:
        return pickle.load(f)

def main():
    inputF = np.array(loadVectorSum('file_name')) # 入力のベクトル
    sugDocHigh = np.array(loadVectorSum('file_name')) # 適合報告書のベクトル和
    sugDocLow = np.array(loadVectorSum('file_name')) # 非適合報告書のベクトル和

    # 主成分分析する
    pca = PCA()
    pca.fit(inputF)
    pca1 = PCA(n_components=2)
    pca1.fit(sugDocHigh)
    pca2 = PCA(n_components=2)
    pca2.fit(sugDocLow)

    # 分析結果を元にデータセットを主成分に変換する
    transformed = pca.fit_transform(inputF)
    transformed1 = pca1.fit_transform(sugDocHigh)
    transformed2 = pca2.fit_transform(sugDocLow)

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
    main()
