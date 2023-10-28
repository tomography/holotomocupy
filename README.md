# holotomo
Holotomographic reconstruction on GPU

see ? for details

## 1. create conda environment and install dependencies

```console
conda create -n holotomo -c conda-forge cupy swig scikit-build dxchange xraylib
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


