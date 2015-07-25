
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

    def cdf(self, x):
        """
        returns cdf at x for each X_i (that is... k values)
        """

        return (self.losses<=x).sum(axis=0)/self._n

    def cdf_of_Sc(self, x):
        """
        returns cdf at x for S
        """
        a = np.array([self.quantile(p/100).sum() for p in range(101)])
        index2 = (a >= self.K).argmax()
        index1 = index2-1
        a = np.array([self.quantile(index1/100 + i*((index2-index1)/100)/10000).sum() for i in range(10001)])
        index = (a>=self.K).argmax()
        return index1/100 + index*((index2-index1)/100)/10000
        #
        # a = np.array([self.quantile(p/10000).sum() for p in range(10000)])
        # index = (a>=self.K).argmax()
        # return index/10000
        # return (self.S<=x).sum()/self._n

    def quantile(self, p, alpha=1):
        """
        p: between 0 and 1
        """
        lower = scoreatpercentile(self.sorted_losses, p*100, axis=0, interpolation_method='lower')
        higher = scoreatpercentile(self.sorted_losses, p*100, axis=0, interpolation_method='higher')
        return alpha*lower+(1-alpha)*higher

    # def quantile_of_S(self, p, alpha=1):
    #     """
    #     p: between 0 and 1
    #     """
    #     lower = scoreatpercentile(self.S, p*100, interpolation_method='lower')
    #     higher = scoreatpercentile(self.S, p*100, interpolation_method='higher')
    #     return alpha*lower+(1-alpha)*higher
    def quantile_of_Sc(self, p, alpha=1):
        """
        p: between 0 and 1
        """
        return self.quantile(p, alpha).sum()



class AllocRule:
    def __init__(self, *args, **kwargs):
        """
        K is the capital to be distributed,
        losses is a numpy array - a column for each line,
        xi - value of a random variable for each observation,
        w - weights
        """
        self.data = Losses(*args, **kwargs)

    def _cdf(self):
        pass

    def _rule(self):
        pass
        
    def _allocate(self):
        pass


class QuantileRule(AllocRule):
    def _find_alpha(self):
        K=self.data.K
        p = self.data.cdf_of_Sc(K)
        a = self.data.quantile_of_Sc(p, 1)
        b = self.data.quantile_of_Sc(p, 0)
        if a == b:  # a+(1-alfa)(b-a)=K .. K-a = (1-alfa)(b-a) .. 1-alfa = (K-a)/(b-a) .. alfa = 1-(K-a)/(b-a)
            return 1
        else:
            return 1-(K-a)/(b-a) # 1-alfa = (K-1)/(b-a) .. (1-alfa)(b-a)=K-1 .. K = 1+(1-alfa)(b-a) .. K=1+b-a+alfa*a-alfa*b = a+

    def allocate(self):
        self.allocations = self.data.quantile(self.data.cdf_of_Sc(self.data.K), self._find_alpha())
        return self.allocations


class QuadraticRule(AllocRule):
    def allocate(self):
        means = self.data.losses.T.dot(self.data.xi)/self.data._n
        self.allocations = means + self.data.w*(self.data.K-means.sum())
        return self.allocations






