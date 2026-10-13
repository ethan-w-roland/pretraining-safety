"""Checks for Trinity scaling, schedules, and expert-independent optimizer state."""

import logging
import unittest
from unittest.mock import patch

import torch

from src.model.config import ModelConfig, RoutedModelConfig, model_config
from src.model.moe import MoETransformer
from src.run.experiment.config import trinity_train_config
from src.run.train.base import BaselineConfig, make_scheduler
from src.run.train.optim import PretrainingOptimizer
from src.run.train.routing import RoutingConfig, do_routing, routing_masks
from src.run.util.config import DataConfig, EvalConfig, ExperimentConfig, TrainConfig


class PretrainingOptimizerTests(unittest.TestCase):
    def test_preset_scaling(self):
        for size, batch, lr in (("25M", 128, 6.75e-4), ("50M", 256, None), ("100M", 512, 4.5e-4)):
            train = trinity_train_config(model_config(size), world_size=2)
            self.assertEqual(train.bs * train.acc_steps * 2, batch)
            self.assertEqual(train.bs, 8)
            self.assertEqual(train.muon_lr, 1e-3)
            self.assertEqual(train.warmup_steps, 2000)
            self.assertEqual(train.min_lr_ratio, 0.0)
            if lr is not None:
                self.assertAlmostEqual(train.lr, lr)

    def test_wsd_boundaries_and_short_runs(self):
        for steps in (1, 2, 100):
            parameter = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.AdamW([parameter], lr=1e-3)
            stage = BaselineConfig(model=ModelConfig(), train=TrainConfig(
                warmup_prc=.1, warmup_steps=2000, decay_prc=.1, min_lr_ratio=0.0,
            ))
            scheduler = make_scheduler(stage, optimizer, steps, logging.getLogger())
            lrs = [scheduler.get_last_lr()[0]]
            for _ in range(steps):
                optimizer.step()
                scheduler.step()
                lrs.append(scheduler.get_last_lr()[0])
            self.assertEqual(lrs[-1], 0.0)
            self.assertTrue(all(0 <= lr <= 1e-3 for lr in lrs))
            if steps == 100:
                self.assertEqual(lrs[0], 0.0)
                self.assertEqual(lrs[10], 1e-3)
                self.assertEqual(lrs[90], 1e-3)
                self.assertAlmostEqual(lrs[95], 5e-4)

    def test_independent_expert_updates_and_idle_state(self):
        torch.manual_seed(1)
        cfg = RoutedModelConfig.from_base(ModelConfig(
            vocab_size=32, embed_dim=16, num_layers=1, num_heads=2,
            num_key_value=1, head_dim=8, mlp_dim=128, ctx_len=8,
        ), arch="moe", core_param_prc=.5, aux_param_prc=.5)
        model = MoETransformer(cfg, ["core", "aux"], dtype=torch.bfloat16)
        train = TrainConfig(lr=6.75e-4, muon_lr=1e-3)
        optimizer = PretrainingOptimizer(model, train)
        self.assertEqual(sum(u.master.numel() for u in optimizer.updates), sum(p.numel() for p in model.parameters()))
        self.assertTrue(all(u.master.dtype == torch.float32 for u in optimizer.updates))
        self.assertEqual(optimizer.optimizers[0].param_groups[0]["adjust_lr_fn"], "original")
        tokens = torch.randint(0, 32, (2, 8))

        # Populate core momentum before an auxiliary-only optimizer step.
        masks = routing_masks(torch.zeros(2, dtype=torch.long), 2)
        model(tokens, targets=tokens, **masks)[1].backward()
        optimizer.step(torch.tensor([1, 0]))
        optimizer.zero_grad()
        before = [u.master.detach().clone() for u in optimizer.updates]
        momentum = {
            id(u.master): optimizer.optimizers[0].state[u.master]["momentum_buffer"].clone()
            for u in optimizer.updates if u.master in optimizer.optimizers[0].state
        }

        masks = routing_masks(torch.ones(2, dtype=torch.long), 2)
        model(tokens, targets=tokens, **masks)[1].backward()
        self.assertTrue(all(p.grad is not None and p.grad.count_nonzero() == 0 for name, p in model.named_parameters() if ".mlp." not in name))
        optimizer.step(torch.tensor([0, 1]))
        aux_changed = False
        for u, previous in zip(optimizer.updates, before):
            if u.expert == 0:
                self.assertTrue(torch.equal(u.master, previous))
                if id(u.master) in momentum:
                    self.assertTrue(torch.equal(optimizer.optimizers[0].state[u.master]["momentum_buffer"], momentum[id(u.master)]))
            else:
                aux_changed |= not torch.equal(u.master, previous)
        self.assertTrue(aux_changed)

    def test_muon_slices_match_separate_matrix_updates(self):
        torch.manual_seed(3)
        cfg = RoutedModelConfig.from_base(ModelConfig(
            vocab_size=32, embed_dim=16, num_layers=1, num_heads=2,
            num_key_value=1, head_dim=8, mlp_dim=128, ctx_len=8,
        ), arch="moe", core_param_prc=.5, aux_param_prc=.5)
        model = MoETransformer(cfg, ["core", "aux"], dtype=torch.float32)
        train = TrainConfig()
        optimizer = PretrainingOptimizer(model, train)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        references = []
        for u in optimizer.updates:
            if u.index is not None:
                p = torch.nn.Parameter(u.master.detach().clone())
                p.grad = u.source.grad[u.index].clone()
                opt = torch.optim.Muon([p], lr=train.muon_lr, weight_decay=train.weight_decay, adjust_lr_fn="original")
                opt.step()
                references.append((u, p))
        optimizer.step(torch.tensor([1, 1]))
        for u, p in references:
            torch.testing.assert_close(u.master, p)
            torch.testing.assert_close(u.source[u.index], p)

    def test_mixed_batch_matches_per_sequence_gradient_reference(self):
        torch.manual_seed(7)
        cfg = RoutedModelConfig.from_base(ModelConfig(
            vocab_size=32, embed_dim=16, num_layers=2, num_heads=2,
            num_key_value=1, head_dim=8, mlp_dim=128, ctx_len=8,
        ), arch="moe", core_param_prc=.5, aux_param_prc=.5)
        model = MoETransformer(cfg, ["core", "a", "b"], dtype=torch.float32)
        tokens = torch.randint(0, 32, (3, 8))
        targets = torch.randint(0, 32, (3, 8))
        labels = torch.tensor([0, 1, 2])
        logits, loss = model(tokens, targets=targets, **routing_masks(labels, 3))
        loss.backward()
        mixed_grads = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        reference_logits = []
        for i in range(3):
            output, sequence_loss = model(tokens[i:i+1], targets=targets[i:i+1], **routing_masks(labels[i:i+1], 3))
            reference_logits.append(output)
            (sequence_loss / 3).backward()
        torch.testing.assert_close(logits, torch.cat(reference_logits), atol=1e-6, rtol=1e-5)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.grad, mixed_grads[name], atol=1e-6, rtol=1e-5)

    def test_training_calls_forward_once_per_loader_batch(self):
        cfg = RoutedModelConfig.from_base(ModelConfig(
            vocab_size=32, embed_dim=16, num_layers=1, num_heads=2,
            num_key_value=1, head_dim=8, mlp_dim=128, ctx_len=8,
        ), arch="moe", core_param_prc=.5, aux_param_prc=.5)
        model = MoETransformer(cfg, ["core", "a", "b"], dtype=torch.float32)
        batches = [
            (torch.tensor([0, 1, 2]), torch.randint(0, 32, (3, 8)), torch.randint(0, 32, (3, 8)))
            for _ in range(2)
        ]

        class Loader:
            def __len__(self):
                return len(batches)

            def __iter__(self):
                return iter(batches)

            def reset(self, epoch):
                pass

        calls = []
        model.register_forward_pre_hook(lambda module, args, kwargs: calls.append(kwargs["tokens"].shape), with_kwargs=True)
        stage = RoutingConfig(model=cfg, train=TrainConfig(bs=3, ctx_len=8), eval=EvalConfig(num_train_evals=0))
        config = ExperimentConfig(device=torch.device("cpu"), data=DataConfig(aux=["a", "b"]))
        with patch("src.run.train.base.make_loader", return_value=Loader()), patch("src.run.train.base.save_model"), patch("src.run.train.base.save_losses"):
            do_routing(stage, model, config)
        self.assertEqual(calls, [torch.Size([3, 8]), torch.Size([3, 8])])


if __name__ == "__main__":
    unittest.main()
