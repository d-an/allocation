The package implements some capital allocation rules following literature mentioned below. 

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
