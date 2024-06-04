# holotomo
Holotomographic reconstruction on GPU


## 1. create conda envi`ronment and install dependencies

```console
conda create -n holotomo -c conda-forge cupy dxchange xraylib matplotlib jupyter astropy olefile
```

## 2. clone the package and install it

```console
git clone https://github.com/nikitinvv/holotomo

cd holotomo

pip install .
```

## 3. See jupyter notebook for examples of reconstructions

*3d_ald_syn/modeling.ipynb* - generate data for a 3D ALD synthetic sample 

*3d_ald_syn/rec.ipynb* - reconstruction of the 3D ALD sample with probe retrieval


