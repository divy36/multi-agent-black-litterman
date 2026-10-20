"""Regression checks for the portable release."""
import numpy as np
import pytest
import simulation_loop_postfix as simulation
from fomc_signals import FOMCSignalExtractor
from performance_metrics import bootstrap_sharpe_ci


@pytest.mark.parametrize('range_text,low,high', [('3.00 to 3.25',3,3.25),('1/4 to 1/2',.25,.5),('2-1/4 to 2-1/2',2.25,2.5)])
def test_rate_ranges(range_text,low,high):
    signal=FOMCSignalExtractor().extract('The target range is '+range_text+' percent.')
    assert signal.rate_found
    assert signal.rate_level_low == low
    assert signal.rate_level_high == high


def test_live_input_required(monkeypatch):
    monkeypatch.setattr(simulation,'load_fomc_text',lambda *args: '')
    with pytest.raises(ValueError,match='source document'):
        simulation.checked_fomc_text('missing.txt','2020-01-01',False)
    assert simulation.checked_fomc_text('missing.txt','2020-01-01',True)==''


def test_nonzero_risk_free_configuration():
    r=np.array([.01,.03,-.02,.005])
    excess=r-((1.04)**(1/12)-1)
    actual=bootstrap_sharpe_ci(r,n_boot=20,risk_free_annual=.04)[0]
    assert actual == pytest.approx(excess.mean()/excess.std(ddof=1)*np.sqrt(12))
