#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:40:41 2016

@author: markno1
"""

from sklearn import datasets
from plot import plotting_grid, plot_info
digits = datasets.load_digits()
from tabulate import tabulate

# plt.imshow(digits.images[5])
# plt.show()


X = digits.data
y = digits.target

X = X[y < 5]
y = y[y < 5]

from sklearn import preprocessing
X = preprocessing.scale(X)


from sklearn.decomposition import PCA
clf = PCA(n_components=2)
X_t = clf.fit_transform(X)


from sklearn.cluster import KMeans
from purity import purity_score

# Plotter
my_plt = plotting_grid(fig_r=22, fig_c=22, grid_r=4, grid_c=4)
table_info = []
plotinfo = []
for i in range(3, 11):

    kmeans = KMeans(i)
    kmeans.fit(X_t)

    my_plt.plot_2D_decision_regions_Kmean(X_t, kmeans, 0.2, "K-mean k= " + str(i), i - 3)

    from sklearn.metrics import normalized_mutual_info_score, homogeneity_score
    norm_mutual = normalized_mutual_info_score(y, kmeans.predict(X_t))
    hom_geneity = homogeneity_score(y, kmeans.predict(X_t))

    table_info.append((i, norm_mutual, hom_geneity, purity_score(kmeans.predict(X_t), y)))
    plotinfo.append((norm_mutual, hom_geneity, purity_score(kmeans.predict(X_t), y)))

print(tabulate(table_info, headers=['K', 'Normalized mutual', 'Homogeneity', 'Purity']))
my_plt.save("Kmeans1")
my_plt.show()
plot_info(plotinfo, "KmeansInfo", 3)

from sklearn import mixture
table_info = []
plotinfo = []

my_plt = plotting_grid(fig_r=22, fig_c=22, grid_r=4, grid_c=4)

for i in range(2, 11):
    mixture.GaussianMixture

    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=i, covariance_type='full').fit(X_t)
    my_plt.plot_2D_decision_regions_GMM(
        X_t, gmm, 0.2, "Gaussian Mixture component = " + str(i), i - 2)
    norm_mutual = normalized_mutual_info_score(y, gmm.predict(X_t))
    hom_geneity = homogeneity_score(y, gmm.predict(X_t))
    table_info.append((i, norm_mutual, hom_geneity, purity_score(gmm.predict(X_t), y)))
    plotinfo.append((norm_mutual, hom_geneity, purity_score(gmm.predict(X_t), y)))
    print(gmm.score(X_t, y))

print(tabulate(table_info, headers=[
      'K', 'Normalized mutual', 'Homogeneity', 'Purity']))
my_plt.save("GMM1")
my_plt.show()
plot_info(plotinfo, "GMMinfo", 2)
