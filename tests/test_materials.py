"""Constitutive model checks against closed-form Mohr-Coulomb results."""

import math

import numpy as np

from lythos.core.materials import (
    LinearElastic, MaterialState, MohrCoulomb, concrete_modulus, elastic_matrix,
    principal_stresses, from_principal,
)


def _plane_strain_compression(mat, confining, n_steps=600, d_eps=-2.0e-4):
    """Strain-controlled compression in y with sigma_xx held constant.

    Returns the stress history.  The lateral strain is solved at each step so
    that the horizontal stress stays at the confining pressure, which is the
    plane-strain equivalent of a stress-controlled triaxial test.
    """
    state = MaterialState.zeros(1)
    state.stress[:] = [confining, confining, confining, 0.0]
    history = []
    exx = 0.0
    for _ in range(n_steps):
        # Newton on the lateral strain increment to keep sigma_xx constant.
        d_exx = 0.0
        for _it in range(30):
            de = np.array([[d_exx, d_eps, 0.0, 0.0]])
            stress, tangent, _ = mat.update(state, de)
            residual = stress[0, 0] - confining
            if abs(residual) < 1e-8 * max(abs(confining), 1.0):
                break
            k = tangent[0, 0, 0]
            d_exx -= residual / (k if abs(k) > 1e-9 else 1.0)
        de = np.array([[d_exx, d_eps, 0.0, 0.0]])
        stress, _, state = mat.update(state, de)
        exx += d_exx
        history.append(stress[0].copy())
    return np.array(history)


def test_mohr_coulomb_failure_deviator():
    """Peak deviator stress must match sigma_1 = sigma_3 Kp + 2 c sqrt(Kp)."""
    c, phi = 10.0, 30.0
    mat = MohrCoulomb(E=2.0e4, nu=0.3, c=c, phi=phi, psi=0.0)
    confining = -100.0                       # 100 kPa compression, tension positive
    hist = _plane_strain_compression(mat, confining)
    s_yy, s_xx = hist[-1, 1], hist[-1, 0]
    q = abs(s_yy - s_xx)

    kp = (1 + math.sin(math.radians(phi))) / (1 - math.sin(math.radians(phi)))
    s1_expected = 100.0 * kp + 2 * c * math.sqrt(kp)
    q_expected = s1_expected - 100.0
    assert abs(q - q_expected) / q_expected < 5e-3, (q, q_expected)


def test_tresca_failure_deviator():
    """With phi = 0 the deviator at failure is 2 su."""
    su = 40.0
    mat = MohrCoulomb(E=1.0e4, nu=0.495, c=su, phi=0.0, psi=0.0)
    hist = _plane_strain_compression(mat, -150.0)
    q = abs(hist[-1, 1] - hist[-1, 0])
    assert abs(q - 2 * su) / (2 * su) < 5e-3, q


def test_yield_surface_never_violated():
    """No returned stress state may sit outside the yield surface."""
    rng = np.random.default_rng(7)
    mat = MohrCoulomb(E=3.0e4, nu=0.3, c=8.0, phi=32.0, psi=5.0)
    state = MaterialState.zeros(400)
    state.stress[:] = rng.normal(-80.0, 60.0, size=(400, 4)) * np.array([1, 1, 1, 0.3])
    for _ in range(25):
        de = rng.normal(0.0, 3.0e-4, size=(400, 4))
        de[:, 2] = 0.0
        stress, _, state = mat.update(state, de)
        s_a, s_b, szz, _, _ = principal_stresses(stress)
        pr = np.sort(np.stack([s_a, s_b, szz], axis=1), axis=1)[:, ::-1]
        sphi = math.sin(math.radians(mat.phi))
        f = (pr[:, 0] - pr[:, 2]) + (pr[:, 0] + pr[:, 2]) * sphi - 2 * mat.c * math.cos(math.radians(mat.phi))
        assert f.max() < 1e-4 * mat.E * 1e-3 + 1e-6, f.max()
        assert pr[:, 0].max() < mat.tension_limit() + 1e-6


def test_elastic_step_reproduces_hookes_law():
    # No tension cut-off and a huge cohesion: the model must reduce to Hooke.
    mat = MohrCoulomb(E=5.0e4, nu=0.25, c=1.0e6, phi=0.0, tension_cutoff=None)
    state = MaterialState.zeros(3)
    de = np.array([[1e-5, -2e-5, 0.0, 3e-5],
                   [-4e-5, 1e-5, 0.0, -1e-5],
                   [0.0, 0.0, 0.0, 2e-5]])
    stress, tangent, _ = mat.update(state, de)
    D = elastic_matrix(mat.E, mat.nu)
    assert np.allclose(stress, de @ D.T, rtol=1e-10, atol=1e-8)
    assert np.allclose(tangent, D, rtol=1e-8, atol=1e-6)


def test_consistent_tangent_matches_finite_difference():
    """The algorithmic tangent must match a numerical derivative of the update."""
    mat = MohrCoulomb(E=2.0e4, nu=0.3, c=10.0, phi=30.0, psi=10.0)
    state = MaterialState.zeros(1)
    state.stress[:] = [-120.0, -180.0, -140.0, -25.0]
    base_de = np.array([[2e-4, -6e-4, 0.0, 1e-4]])
    stress0, tangent, _ = mat.update(state, base_de)

    num = np.zeros((4, 4))
    h = 1e-9
    for j in (0, 1, 3):
        de = base_de.copy()
        de[0, j] += h
        s_plus, _, _ = mat.update(state, de)
        num[:, j] = (s_plus[0] - stress0[0]) / h
    err = np.abs(num[:, [0, 1, 3]] - tangent[0][:, [0, 1, 3]]).max()
    assert err < 0.02 * mat.E, err


def test_strength_reduction_scales_c_and_tan_phi():
    mat = MohrCoulomb(c=20.0, phi=30.0, psi=10.0)
    red = mat.reduced(1.5)
    assert abs(red.c - 20.0 / 1.5) < 1e-12
    assert abs(math.tan(math.radians(red.phi)) - math.tan(math.radians(30.0)) / 1.5) < 1e-12
    assert red.psi <= red.phi


def test_concrete_modulus_en1992():
    # EN 1992-1-1 Table 3.1 rounds Ecm to the nearest GPa, so compare loosely.
    assert abs(concrete_modulus(30.0) - 33.0e6) < 0.5e6      # C30/37 -> 33 GPa
    assert abs(concrete_modulus(25.0) - 31.0e6) < 0.5e6      # C25/30 -> 31 GPa
    assert abs(concrete_modulus(40.0) - 35.0e6) < 0.5e6      # C40/50 -> 35 GPa


def test_principal_round_trip():
    rng = np.random.default_rng(3)
    sig = rng.normal(size=(50, 4))
    s_a, s_b, szz, c2, s2 = principal_stresses(sig)
    assert np.all(s_a >= s_b - 1e-12)
    assert np.allclose(from_principal(s_a, s_b, szz, c2, s2), sig, atol=1e-12)
