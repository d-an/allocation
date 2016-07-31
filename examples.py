# nacti potrebne knihovny
import numpy as np
import allocation as allocs
from scipy.stats import scoreatpercentile, norm
import pandas as pd
import matplotlib.pyplot as plt



# vygenerovani dat:
np.random.seed(112233)
data = np.hstack((np.random.lognormal(1, 1, 20),
                  np.random.lognormal(2, 1, 20),
                  np.random.lognormal(3, 2, 20))).reshape(-1,3)

# vytvoreni datoveho objektu pro praci s alok. pravidly
l = allocs.Losses(data, 6)


##
# alpha vs alphas:
# alpha:
rule = allocs.QuantileRule(l)
p = rule._cdf_of_Sc(rule.data.K)
deecka = rule._quantile(p, 0) - rule._quantile(p, 1)
rule.allocate()
pd.Series(rule._quantile(p, 0)).plot(kind='bar')
rule.plot()
allocations_alpha = rule.allocations
left_quantiles = rule._quantile(p, 1)
pd.Series(left_quantiles).plot(kind='bar')
results = pd.DataFrame(np.array([left_quantiles, allocations_alpha, deecka,
                                 allocations_alpha - left_quantiles]).T,
             columns=['left quantiles:', 'with one alpha:', 'deecka:', 'difference'])
print(results)
plt.show()




# alphas:
pd.Series(rule._quantile(p, 0)).plot(kind='bar')
rule.allocate(alphas=True).plot()
left_quantiles = rule._quantile(p, 1)
pd.Series(left_quantiles).plot(kind='bar')
allocations_alpha = rule.allocations
results = pd.DataFrame(np.array([left_quantiles, allocations_alpha, deecka,
                                 allocations_alpha - left_quantiles]).T,
             columns=['left quantiles:', 'with one alpha:', 'deecka:', 'difference'])

print(results)




# mnohorozmerne normalni rozdeleni
# vygenerovani dat
np.random.seed(123)
sigma = np.array([[1,0.2,0.5],
                  [0.2,1,0.1],
                  [0.5,0.1,1]])
data = np.random.multivariate_normal([1,2,3], sigma, 1000)



##### CTE:
# presne: 
p = 0.95
sigma1 = 1
sigma2 = 1
sigma3 = 1
sigma12 = 0.2
sigma13 = 0.5
sigma23 = 0.1
sigmaS = sigma.sum()
sigmakS = sigma.sum(axis=1)

C = norm.pdf(norm.ppf(p))/(1-p)
K = norm(6, np.sqrt(sigmaS)).ppf(p)
alokace = np.array([C*s/np.sqrt(sigmaS) for s in sigmakS])
alokace = alokace + np.array([1,2,3])
gamma = alokace.sum()
alokace_CTE = K/gamma*alokace
print(alokace_CTE)


# odhadnute z dat:
K = norm(6, np.sqrt(sigmaS)).ppf(p)
l = allocs.Losses(data, K)
rule = allocs.TailvarRule(l)
rule.allocate(p=p)


print(np.sum(alokace_CTE), np.sum(rule.allocations))
print('podily', rule.allocations/alokace_CTE)



##### haircut alokace
# presne: 
kvantil1 = norm(1, 1).ppf(p)
kvantil2 = norm(2, 1).ppf(p)
kvantil3 = norm(3, 1).ppf(p)
C = kvantil1+kvantil2+kvantil3
K = norm(6, np.sqrt(sigmaS)).ppf(p)
kvantily = [kvantil1, kvantil2, kvantil3]
alokace = [K/C*kv for kv in kvantily]
alokace_haircut = alokace
print(alokace)


# odhadnute z dat:
K = norm(6, np.sqrt(sigmaS)).ppf(p)
l = allocs.Losses(data, K)
rule = allocs.HaircutRule(l)
rule.allocate(p=p)

print(np.sum(alokace), np.sum(rule.allocations))
print('podily', rule.allocations/np.array(alokace))



####
# kovariancni alokace
# presne: 
covs = sigma.sum(axis=0)
C = sigma.sum()
K = norm(6, np.sqrt(sigmaS)).ppf(p)
alokace = [K/C*covariance for covariance in covs]
alokace_kovariancni = alokace
print(alokace)


# odhadnute z dat:
K = norm(6, np.sqrt(sigmaS)).ppf(p)
l = allocs.Losses(data, K)
rule = allocs.CovarianceRule(l)
rule.allocate()


print(np.sum(alokace), np.sum(rule.allocations))
print('podily', rule.allocations/np.array(alokace))



### podily alokaci jednotlivych pravidel:
print(np.array(alokace_haircut)/np.array(alokace_CTE))
print(np.array(alokace_kovariancni)/np.array(alokace_CTE))
print(np.array(alokace_kovariancni)/np.array(alokace_haircut))


