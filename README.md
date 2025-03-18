# ERPCA: Robust Principal Component Analysis for Exponential Family Distributions


Robust Principal Component Analysis for Exponential Family Distributions (eRPCA) is a method for jointly recovering embedded low-rank structures and corresponding sparse anomalies from data matrices corrupted by non-Gaussian noise from the exponential family distribution This package implements eRPCA in both single group setting and mutil-group setting. The implementation refers [the paper][Lin] and its [R implementation][R Implementation].


[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


## Installation  


To install the package from repository:  


```sh
pip install git+https://github.com/FYlee39/ERPCA_Python.git@v1.0.0
```

## Usage


```python
import numpy as np

from eRPCA_py import eRPCA

# single group
data_sample = np.random.random((20, 20, 20))
erpca = eRPCA.ERPCA(observation_matrix=data_sample)
L_single, S_single = erpca.run()

# multi-group
data_sample_group = np.random.random((20, 20, 20, 2))
erpca_group = eRPCA.ERPCA(observation_matrix=data_sample_group)
L_group, S_group = erpca_group.run()
```

## Contributing

Contributing is welcome!


## References

- [X. Zheng, S. Mak, L. Xie, and Y. Xie. eRPCA: Robust Principal Component Analysis for Exponential Family Distributions. Statistical Analysis and Data Mining: An ASA Data Science Journal, 7(2):e11670, 2024. doi: https://doi.org/10.1002/sam.11670.][Lin]
- [R Implementation]

[R Implementation]: https://github.com/Xiaojzheng/ERPCA
[Lin]: https://onlinelibrary.wiley.com/doi/10.1002/sam.11670


