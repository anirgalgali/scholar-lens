import numpy
import torch
from src.mintransformer.functional import gelu,silu,softmax
import torch.nn.functional as F

def test_gelu():

    x = torch.randn(
        100,
    )
    reference_gelu_out = torch.nn.functional.gelu(x, approximate="none")
    my_gelu_out = gelu(x)
    assert torch.allclose(my_gelu_out, reference_gelu_out, atol=1e-6)

    reference_gelu_out_approx = torch.nn.functional.gelu(x, approximate="tanh")
    my_gelu_out_approx = gelu(x, approximate="tanh")
    assert torch.allclose(my_gelu_out_approx, reference_gelu_out_approx, atol=1e-6)

def test_silu():
    x = torch.randn(100,)
    reference_silu_out = torch.nn.functional.silu(x)
    my_silu_out = silu(x)
    assert torch.allclose(my_silu_out, reference_silu_out, atol=1e-6)

def test_softmax_matches_pytorch():
    x = torch.tensor(
        [
            [0.4655, 0.8303, 0.9608, 0.9656, 0.6840],
            [0.2583, 0.2198, 0.9334, 0.2995, 0.1722],
            [0.1573, 0.6860, 0.1327, 0.7284, 0.6811],
        ]
    )
    expected = F.softmax(x, dim=-1)
    numpy.testing.assert_allclose(
        softmax(x, dim=-1).detach().numpy(), expected.detach().numpy(), atol=1e-6
    )
    # Test that softmax handles numerical overflow issues
    numpy.testing.assert_allclose(
        softmax(x + 100, dim=-1).detach().numpy(),
        expected.detach().numpy(),
        atol=1e-6,
    )