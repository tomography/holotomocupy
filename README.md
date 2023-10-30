# holotomo
Holotomographic reconstruction on GPU


## 1. create conda environment and install dependencies

```console
conda create -n holotomo -c conda-forge cupy swig cmake scikit-build dxchange xraylib matplotlib jupyter astropy olefile
```

Note: CUDA drivers need to be installed before installation

## 2. install

```console
pip install .
```

## 3. check adjoint tests

```console
cd tests

```

Adjoint test holography

```console
python test_holo.py

```

Adjoint test tomography

```console
python test_tomo.py

```

See jupyter notebook for examples of reconstructions



DTU installation and run jupyter notebook


```console
voltash -X

module add cuda/12.2

conda create -n holotomo -c conda-forge cupy swig cmake scikit-build dxchange xraylib matplotlib jupyter astropy olefile

conda activate holtomo

git clone https://github.com/nikitinvv/holotomo

cd holotomo

pip install .

cd tests

pyton test_holo.py

jupyter notebook

```


