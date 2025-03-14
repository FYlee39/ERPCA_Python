'''
eRPCA_py: Python implementation of eRPCA.

The exponential family RPCA (eRPCA) method, introduced in Zheng et al. [1], extends
RPCA to data following an exponential family distribution. This package implements eRPCA in Python.

References
----------
The package implements Algorithm 1, 2 & 3 of the paper [1]_.

.. [1] X. Zheng, S. Mak, L. Xie, and Y. Xie.
    eRPCA: Robust Principal Component Analysis for Exponential Family Distributions.
    Statistical Analysis and Data Mining: An ASA Data Science Journal, 17(2):e11670, 2024.
    doi: https://doi.org/10.1002/sam.11670.
    URL: https://onlinelibrary.wiley.com/doi/pdf/10.1002/sam.11670.

Examples
--------
>>> import numpy as np
>>> import eRPCA_py.eRPCA
>>>
>>> data_sample = np.random.random((20, 20, 20))
>>> erpca = eRPCA_py.eRPCA.ERPCA(observation_matrix=data_sample)
>>> L_est, S_est = erpca.run()
'''