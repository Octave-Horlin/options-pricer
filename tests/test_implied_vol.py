import numpy as np
import pytest
from black_scholes import bs_call, bs_put
from implied_vol import implied_vol

S, K, T, r = 100.0, 100.0, 1.0, 0.05
TOL = 1e-6


@pytest.mark.parametrize("sigma_true", [0.05, 0.10, 0.20, 0.40, 0.80])
@pytest.mark.parametrize("option_type", ['call', 'put'])
@pytest.mark.parametrize("moneyness", [0.80, 0.90, 1.00, 1.10, 1.20])
def test_round_trip(sigma_true, option_type, moneyness):
    K_ = S / moneyness
    pricer = bs_call if option_type == 'call' else bs_put
    price = pricer(S, K_, T, r, sigma_true)
    recovered = implied_vol(price, S, K_, T, r, option_type)
    assert not np.isnan(recovered), f"implied_vol returned NaN for sigma={sigma_true}, type={option_type}, K={K_:.1f}"
    assert abs(recovered - sigma_true) < TOL


@pytest.mark.parametrize("option_type,bad_price", [
    ('call', 0.0),           # at lower arbitrage bound (should give NaN)
    ('call', 200.0),         # above spot (upper bound violation)
    ('put',  0.0),           # below intrinsic
    ('put',  100.0),         # above discounted strike
])
def test_out_of_bounds_returns_nan(option_type, bad_price):
    result = implied_vol(bad_price, S, K, T, r, option_type)
    assert np.isnan(result)
