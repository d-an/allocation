
import numpy as np
from scipy.stats import scoreatpercentile
from pandas import Series
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from itertools import combinations, chain
import pyperclip

class Losses:
    def __init__(self, losses, K=None):
        """
        K is the capital to be distributed,
        losses is a numpy array - a column for each line,
        xi - value of a random variable for each observation,
        w - weights
        """
        self.K = K
        self.losses = losses
        # self.sorted_losses = losses.copy()
        # self.sorted_losses.sort(axis=0)
        self.n, self.k = self.losses.shape
        self.S = losses.sum(axis=1)
        # self.S.sort()

    def __repr__(self):
        return '<Losses: K={}, losses=array of shape {}>'.format(self.K, self.losses.shape)

    def __getitem__(self, item):
        l = self.losses[item]
        if len(l.shape) == 1:
            l = l.reshape(1, -1)
        return Losses(l, self.K)


########################
#  allocation rules:
########################

"""
alokacni pravidlo ma metodu allocate a jedine, co dela, je, ze vraci alokace
"""


class AllocRule(metaclass=ABCMeta):
    def __init__(self, data):
        """
        K is the capital to be distributed,
        losses is a numpy array - a column for each line,
        xi - value of a random variable for each observation,
        w - weights
        """
        self.data = data

    def plot(self, *args):
        s = Series(self.allocations.ravel())
        s.plot(kind='bar')
        plt.xticks(range(self.data.k), range(1, self.data.k + 1), rotation=0)
        #plt.show()

    @abstractmethod
    def allocate(self):
        pass

#    def test_last_row(self, last):
#        # if last.ndim==1:
#        #     last = last.reshape(1, -1)
#        if last.losses.shape[0] != 1:
#            assert False, 'losses for the test should only have one observation.'
#        return np.sqrt(((self.allocations - last.losses)**2).sum())

    def __repr__(self):
        return '<{}: data={}>'.format(self.__class__.__name__, self.data.__repr__())
        
    def to_clipboard(self):
        pyperclip.copy('[' + ', '.join([str(item) for item in self.allocations]) + ']')


class QuadraticRule(AllocRule):
    def allocate(self, xi=None, w=None):
        proportional = False
        if xi is None:
            self.xi = np.ones((self.data.n, 1))
        else:
            self.xi = xi
        if w is None:
            proportional = True
        else:
            self.w = w.reshape(-1,1)
        means = (self.data.losses * self.xi).mean(axis=0).reshape(-1,1)
        # means = self.data.losses.T.dot(self.xi)/self.data.n
        if proportional:
            gamma = means.sum()
            self.allocations = self.data.K*means/gamma
            print(self.allocations)
            return self
        else:
            self.allocations = means + self.w*(self.data.K-means.sum())
            print('allocations: ', self.allocations)
            return self


#class QuadraticRuleCombined(QuadraticRule):
#    def allocate(self, xi1=None, xi2=None, alpha=1, w=None):
#        if xi1 is None and xi2 is None:
#            return QuadraticRule.allocate(self, w=w)
#        if xi1 is None:
#            return QuadraticRule.allocate(self, xi=xi2, w=w)
#        if xi2 is None:
#            return QuadraticRule.allocate(self, xi=xi1, w=w)
#        xi = alpha*xi1 + (1-alpha)*xi2
#        return QuadraticRule.allocate(self, xi=xi, w=w)


class QuantileRule(AllocRule):
    def _quantile(self, p, alpha=1):
        losses = self.data.losses
        lower = scoreatpercentile(losses, 100*p, axis=0, interpolation_method='lower')
        higher = scoreatpercentile(losses, 100*p, axis=0, interpolation_method='higher')
        return alpha*lower + (1-alpha)*higher

    def _quantile_of_Sc(self, p, alpha=1):
        return self._quantile(p, alpha).sum()

    def _cdf_of_Sc(self, x):
        # = sup{u: g(u)<=x}, g(u) = self.quantile_of_Sc(u)
        values_of_Sc = np.array([self._quantile_of_Sc(u/self.data.n) for u in range(self.data.n+1)])
        index = (values_of_Sc >= x).argmax() - 1
        if index == -1:
            index = 0
        return index/self.data.n

    def _find_alpha(self):
        p = self._cdf_of_Sc(self.data.K)
        self._p = p
        lower = self._quantile_of_Sc(p)
        upper = self._quantile_of_Sc(p, 0)
        self.alpha = 1-(self.data.K - lower)/(upper-lower)
        return self.alpha

    def _additional_allocs(self):
        deecka0 = self._quantile(self._p, 0)-self._quantile(self._p, 1)
        deecka = deecka0.copy()
        m = (deecka > 0).sum()
        additional_allocs = np.zeros(deecka.shape)
        amount = (1-self.alpha)*deecka.sum()
        while amount > 0:
            dmin = deecka[deecka > 0].min()
            if dmin <= amount/m:
                additional_allocs[deecka > 0] += dmin
                deecka[deecka > 0] -= dmin
                amount -= dmin*m
            else:
                additional_allocs[deecka > 0] += amount/m
                deecka[deecka > 0] -= amount/m
                amount = 0
            m = (deecka > 0).sum()
        self.alphas = 1-additional_allocs/deecka0
        return additional_allocs

    def allocate(self, alphas=False):
        self.allocations = self._quantile(self._cdf_of_Sc(self.data.K), self._find_alpha())
        if alphas:
            self.allocations = self._quantile(self._p, 1)+self._additional_allocs()
        print('allocations: ', self.allocations)
        return self


class HaircutRule(AllocRule):
    def _quantile(self, p, alpha=1):
        losses = self.data.losses
        lower = scoreatpercentile(losses, 100*p, axis=0, interpolation_method='lower')
        higher = scoreatpercentile(losses, 100*p, axis=0, interpolation_method='higher')
        return alpha*lower + (1-alpha)*higher

    def allocate(self, p=0.99):
        quantiles = self._quantile(p)
        gamma = quantiles.sum()
        self.allocations = self.data.K*quantiles/gamma
        print('allocations: ', self.allocations)
        return self


class CovarianceRule(AllocRule):
    def allocate(self):
        data = self.data
        n = self.data.n
        covs = data.losses.T.dot(data.S)/n - data.losses.T.mean(axis=1)*data.S.mean()
        gamma = covs.sum()
        self.allocations = data.K*covs/gamma
        self.covs = covs
        print('allocations: ', self.allocations)
        return self


class TailvarRule(AllocRule):
    def allocate(self, p=0.99):
        losses, S = self.data.losses, self.data.S
        s = scoreatpercentile(S, 100*p)
        means = losses[S>s, :].mean(axis=0)
        gamma = means.sum()
        self.allocations = self.data.K*means/gamma
        print('allocations: ', self.allocations)
        return self

class Projection(AllocRule):
    def allocate(self, measure):
        numbers = measure(self.data).measure().risks
        self.allocations = numbers + (self.data.K - numbers.sum())/len(numbers)
        print('allocations: ', self.allocations)
        return self


class Shapley(AllocRule):
    def __init__(self, numbers):
        self.numbers = numbers

    def allocate(self):
        n = len(self.numbers)
        allocations = [0 for i in self.numbers]
        coalitions = chain(*[combinations(self.numbers, i) for i in range(1, n+1)])
        for coalition in coalitions:
            s = len(coalition)
            fraction = 1
            for i in range(s):
                fraction *= (n-i)/(i+1)
            fraction = 1/(s*fraction)
            for member in coalition:
                index = self.numbers.index(member)
                allocations[index] += fraction*member
        self.allocations = np.array(allocations)
        print(self.allocations)
        return self



#def projection(numbers, K):
#    numbers = np.array(numbers)
#    return numbers + (K - numbers.sum())/len(numbers)


########################
#  risk measures:
########################

class RiskMeasure(metaclass=ABCMeta):
    def __init__(self, data):
        """
        losses is a numpy array - a column for each line
        """
        self.data = data

    def plot(self, *args):
        s = Series(self.risks)
        s.plot(kind='bar')
        plt.show()

    @abstractmethod
    def measure(self):
        pass

    def __repr__(self):
        return '<{}: data={}>'.format(self.__class__.__name__, self.data.__repr__())

class VaR(RiskMeasure):
    def measure(self, p=0.99):
        self.risks = scoreatpercentile(losses, 100*p, axis=0, interpolation_method='lower')
        print(self.risks)
        return self


class Covariances(RiskMeasure):
    def measure(self):
        data = self.data
        n = self.data.n
        self.risks = data.losses.T.dot(data.S)/n - data.losses.T.mean(axis=1)*data.S.mean()
        print(self.risks)
        return self


# tests:
if __name__ == '__main__':
    losses = np.random.randn(300).reshape(-1, 3)
    l = Losses(losses, 2.1)

    # choice of xi1:
    p = 0.99
    kvantily = scoreatpercentile(l.losses, 100*p, axis=0, interpolation_method='lower')
    psti = (l.losses > kvantily).sum(axis=0)/l.n
    xi1 = (l.losses > kvantily)/psti  # h_i(X_i)

    # choice of xi2:
    kvantil = scoreatpercentile(l.S, 100*p, interpolation_method='lower')
    pst = (l.S > kvantil).sum()/l.n
    xi2 = ((l.S > kvantil)/pst).reshape(-1,1)
    xi2 = np.hstack((xi2, xi2, xi2))

    # choice of w:
    w=np.ones((l.k, 1))/l.k

    # QuadraticRule:
    print(QuadraticRule(l).allocate(xi=xi1, w=w).allocations.sum())

#    # QuadraticRuleCombined:
#    print(QuadraticRuleCombined(l).allocate(xi1=xi1, xi2=xi2, alpha=0.5, w=w).allocations.sum())

    # QuantileRule:
    print(QuantileRule(l).allocate().allocations.sum())
    print(QuantileRule(l).allocate(alphas=True).allocations.sum())

    # HaircutRule:
    print(HaircutRule(l).allocate().allocations.sum())

    # CovarianceRule:
    print(CovarianceRule(l).allocate().allocations.sum())

    # TailvarRule:
    print(TailvarRule(l).allocate().allocations)

