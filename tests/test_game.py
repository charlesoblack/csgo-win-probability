#! /usr/bin/env python3

import torch
import csgo_wp


class Test_CNN:

    def test_dims_default(self):
        mod = csgo_wp.CNNModel()

        t = torch.rand(size=(5, 1, 10, 10))

        result = mod(t)

        assert result.shape == (5, 2)
