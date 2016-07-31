The package implements some capital allocation rules explained e.g. in the literature mentioned in the references below. 

Installation
------------
To install the package you can use pip: 
`pip install git+git://github.com/d-an/allocation`


How To Use
----------



### Generate Data
```python
np.random.seed(123)
sigma = np.array([[1,0.2,0.5],
                  [0.2,1,0.1],
                  [0.5,0.1,1]])
data = np.random.multivariate_normal([1,2,3], sigma, 1000)
```







Requirements
------------

Written using Python 3.4, Depends on numpy, scipy, matplotlib, pandas and pyperclip (for the copy/paste functionality).

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

