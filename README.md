# ptychotomo
Multi-GPU reconstruction of 3D ptychographictomographic data with non-rigid projection alignment

see ? for details

## 1. create conda environment and install dependencies

```console
conda create -n ptychotomo -c conda-forge cupy swig scikit-build dxchange opencv tomopy silx
```

Note: CUDA drivers need to be installed before installation

## 2. install

```console
python setup.py install
```

## 3. check adjoint tests

```console
cd tests

```

Test deformation

```console
python test_deform.py

```
sample output:

```console
registration time: 5.561098337173462
apply flow time: 0.01982426643371582
data0-data1=5928.0068359375
data0-data1_unwrap=2606.488037109375
norm flow = 10330.0009765625
<data,D*Ddata>=<Ddata,Ddata>: 1.108183e+08 ? 1.164992e+08
```

Test tomography

```console
python test_tomo.py
```

sample output:

```console
norm data = 24286.515625
norm object = 4536712.0
<u,R*Ru>=<Ru,Ru>: 5.897588e+08+8.920990e+01j ? 5.898347e+08+0.000000e+00j
```

Test ptychography

```console
python test_ptycho.py
```

sample output:

```console
np.linalg.norm(psi) = 2.3365922
psi.shape = (174, 256, 256)
psia.shape = (174, 256, 256)
prb.shape = (174, 4, 128, 128)
prba.shape = (174, 4, 128, 128)
np.linalg.norm(data) = 597.29376
np.linalg.norm(psia) = 203.73346
np.linalg.norm(prba) = 10284.095
<psi,Q*F*FQpsi>=<FQpsi,FQpsi>: 3.567599e+05-1.539114e-03j ? 3.567811e+05+0.000000e+00j
<prb,P*F*FPprb>=<FPprb,FPprb>: 3.567594e+05-1.521850e-03j ? 3.567811e+05+0.000000e+00j
```


## 4. test gradient solvers

Test the gradient solver for deformation

```console
python test_deform_grad.py
```

Test the gradient solver for tomography

```console
python test_tomo_grad.py
```

Test the gradient solver for ptychography

```console
python test_ptycho_grad.py
```

## 5. test admm solver for joint ptycho-tomography problem with alignment

```console
python test_admm.py
```
