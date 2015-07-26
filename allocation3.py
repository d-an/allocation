

import numpy as np
from scipy.stats import scoreatpercentile
from statsmodels.distributions.empirical_distribution import ECDF


"""
postup implementace:
Losses - objekt pro ulozeni dat a napocitani zakladnich charakteristik
AllocRule - vzor pro jednotliva pravidla
Allocations - hlavni objekt. zavola se, daji se mu data
            - ulozi data do Losses objektu
            - ma metodu allocate
                - zavola konkretni pravidlo, poskytne mu data
                    - pokud si pravidlo chce neco dopocitat, ulozi to u sebe, necpe to do Losses objektu
                - ulozi alokace, zobrazi je, vrati self
"""


class Losses:
    def __init__(self, K, losses, xi=None, w=None):
        """
        K is the capital to be distributed,
        losses is a numpy array - a column for each line,
        xi - value of a random variable for each observation,
        w - weights
        """
        self.losses = losses
        self.sorted_losses = losses.copy()
        self.sorted_losses.sort(axis=0)
        self.K = K
        self._n, self._k = self.losses.shape
        self.S = losses.sum(axis=1)
        self.S.sort()
        if xi is None:
            self.xi = np.ones((self._n, 1))
        else:
            self.xi = xi
        if w is None:
            self.w = np.ones((self._k, 1))/self._k
        else:
            self.w = w


class QuadraticRule(AllocRule):
    def allocate(self):
        means = self.data.losses.T.dot(self.data.xi)/self.data._n
        self.allocations = means + self.data.w*(self.data.K-means.sum())
        return self.allocations
