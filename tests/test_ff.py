import numpy as np
import pytest
from leadengine.ff import ap_elliptic, EllipticAngleDataset, \
    UniformAngleNull, SatoTateNull
from leadengine.search import evolutionary_search

def test_hasse_bound_holds():
    # RH is a theorem here (Hasse/Weil). If this fails, the pipeline is wrong — full stop.
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = int(rng.choice([101, 211, 307, 401, 499]))
        a, b = int(rng.integers(p)), int(rng.integers(p))
        if (4 * a**3 + 27 * b**2) % p == 0:
            continue
        assert abs(ap_elliptic(a, b, p)) <= 2 * np.sqrt(p)

def test_known_curve_value():
    # y^2 = x^3 + x over F_5 has 4 points (x,y) plus point at infinity at #E=4? No:
    # TODO(agent): compute #E(F_5) for y^2=x^3+x by brute force in the test itself and
    # assert ap_elliptic(1, 0, 5) == 5 + 1 - that count. (Self-verifying, no constants.)
    p = 5
    count = 1
    for x in range(p):
        rhs = (x**3 + x) % p
        for y in range(p):
            count += int((y * y - rhs) % p == 0)
    assert ap_elliptic(1, 0, 5) == p + 1 - count

@pytest.mark.slow
def test_sato_tate_rediscovered_vs_uniform():
    # Known-true deep structure: the engine must distinguish real Frobenius angles
    # from uniform — i.e., rediscover the Sato-Tate law from raw point counts.
    real = EllipticAngleDataset()
    res = evolutionary_search(real, UniformAngleNull(), budget=1500, seed=0)
    assert res.best is not None and res.best.ood_auc >= 0.65

@pytest.mark.slow
def test_nothing_beyond_sato_tate_in_iid_family():
    # Random independent curves carry no structure beyond the ST law: the matched null
    # must absorb everything. This is the deep-structure analogue of the LO-S
    # pair-matched calibration.
    real = EllipticAngleDataset()
    res = evolutionary_search(real, SatoTateNull(), budget=1500, seed=0)
    assert res.best is None or res.best.promoted is False
