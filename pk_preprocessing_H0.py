import numpy as np

from scipy.stats import qmc
sampler = qmc.LatinHypercube(d=1)
sample = sampler.random(n=8000)
h_lhs = qmc.scale(sample, 55, 85)

h_lhs = np.asarray(h_lhs).astype('float32')

import camb
from camb import model, initialpower

Npoints = 500

kmin = 1e-4
kmax = 15

input_pk = []
s8_fid = 0.8102

for i in range(len(h_lhs)):
    #pars.set_cosmology(H0=100 * h_lhs[i])
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h_lhs[i], ombh2=0.02242, omch2=0.11933)
    pars.InitPower.set_params(As=2.105209331e-9, ns=0.9665)
    pars.set_matter_power(redshifts=[0.0], kmax=15.0)
    pars.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')

    # Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, un_pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=Npoints)
    s8_camb = np.array(results.get_sigma8())
    Renorm_Factor = s8_fid**2/s8_camb**2
    pk = Renorm_Factor * un_pk
    input_pk.append(np.asarray(pk[0]).astype('float32'))
    print(i)

input_pk = np.asarray(input_pk).astype('float32')

with open('./input_pk_h0.npy', 'wb') as f:
    np.savez(f, H=h_lhs, k=kh, pk=input_pk)