import numpy as np
import pytest

from qaoa import (
    GROVER_MIXER,
    X_MIXER,
    initial_parameters as seeded_initial_parameters,
    probabilities,
    qaoa_state,
)


def test_seeded_parameters_are_reproducible():
    expected = [1.3789715463539463, 1.4497639358789494]
    assert np.allclose(seeded_initial_parameters(1, 2601), expected)
    assert np.array_equal(
        seeded_initial_parameters(2, 2601),
        seeded_initial_parameters(2, 2601),
    )


@pytest.mark.parametrize("mixer", [X_MIXER, GROVER_MIXER])
@pytest.mark.parametrize("depth", [1, 2])
def test_qaoa_state_is_normalized(cost_diagonal, mixer, depth):
    state = qaoa_state(
        cost_diagonal,
        seeded_initial_parameters(depth, 2601),
        depth=depth,
        mixer=mixer,
    )
    assert state.shape == (2**14,)
    assert probabilities(state).sum() == pytest.approx(1.0, abs=1e-12)
