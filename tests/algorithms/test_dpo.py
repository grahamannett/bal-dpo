import unittest

import torch

from bal_dpo.algorithms.dpo import get_batch_logps, preference_loss


class TestDPO(unittest.TestCase):
    def test_dpo(self):
        policy_chosen_logps = torch.tensor([0.1, 0.2, 0.3])
        polichy_rejected_logps = torch.tensor([0.2, 0.3, 0.4])

        reference_chosen_logps = torch.tensor([0.1, 0.2, 0.3])
        reference_rejected_logps = torch.tensor([0.2, 0.3, 0.4])

        beta = 0.99

        batch_logps = get_batch_logps(
            torch.rand(3, 32, 128),  # batch_size, seq_len, vocab_size
            torch.randint(0, 128, (3, 32)),  # batch_size, seq_len
        )

        losses, chosen_rewards, rejected_rews = preference_loss(
            policy_chosen_logps,
            polichy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta,
        )
