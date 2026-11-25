import pytest
import torch

from ml.patchcore_lite import greedy_coreset


def test_greedy_coreset_is_repeatable_and_bounded() -> None:
    features = torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [8.0, 0.0]])
    first = greedy_coreset(features, 3, seed=3)
    second = greedy_coreset(features, 3, seed=3)

    assert torch.equal(first, second)
    assert first.shape == (3, 2)


@pytest.mark.parametrize("count", [0, 5])
def test_greedy_coreset_rejects_invalid_size(count: int) -> None:
    with pytest.raises(ValueError):
        greedy_coreset(torch.ones((4, 2)), count)
