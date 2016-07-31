This package implements some capital allocation rules explained e.g. in the literature mentioned in the references below. 

Installation
------------
To install the package you can use pip: 
`pip install git+git://github.com/d-an/allocation`


How To Use
----------

There are two types of objects in this package. First, you create an instance of Losses which encapsulates the data you 
have about the losses - *nxk* matrix with *n* observations of losses of *k* lines of business. Then this instance is 
passed to some allocation rule. Object representing an allocation rule can subsequently calculate the allocations, draw
a graph etc. Let's illustrate this with an example:


###### Generate Data
Some imports first:
```python
import numpy as np
import allocation as allocs
import matplotlib.pyplot as plt
from scipy.stats import norm
```

now generate the data:
```python
np.random.seed(123)
sigma = np.array([[1,0.2,0.5],
                  [0.2,1,0.1],
                  [0.5,0.1,1]])
data = np.random.multivariate_normal([1,2,3], sigma, 1000)
```

###### Create the Losses instance
*K* - the overall amount to be allocated between individual risks - can be calculated e.g. as a quantile of the sum
of the individual risks: `K = norm(6, np.sqrt(sigma.sum())).ppf(0.95)`.
```python
l = allocs.Losses(data, K)
```

###### Use an Allocation Rule
Let's use the haircut allocation rule: 
```python
rule = allocs.HaircutRule(l)
rule.allocate()
rule.allocations
rule.to_clipboard()
```
The *allocate* method prints the allocations to the standard output. We can also find the allocations in the *allocations* 
attribute of the *rule* object. The *to_clipboard* method can also be used if we want to paste the values somewhere else. 

###### Draw a Graph
Finally we can use the *plot* method to draw a graph with the allocations. 



Requirements
------------

Written using Python 3.4. Depends on numpy, scipy, matplotlib, pandas and pyperclip (for the copy/paste functionality).

References
----------
```
[1] DHAENE, J. - TSANAKAS, A. - VALDEZ, E. A. - VANDUFFEL, S.,
    Optimal Capital Allocation Principles,
    The Journal of Risk and Insurance, 2012, 
    vol. 79, no. 1, p. 1-28, ISSN 1539-6975.

[2] DENAULT, M., Coherent Allocation of Risk Capital, 
    Journal of Risk, 2001, vol. 4, p. 1–34. ISSN 1465-1211.
          
[3] PANJER, H. H., Measurement of Risk, Solvency Requirements 
    and Allocation of Capital within Financial Conglomerates, 
    Research Report 01-15, University of Waterloo, 2002.
```

