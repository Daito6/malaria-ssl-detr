"""Unit tests for query expansion and the DETR criterion (no model download needed)."""
import torch

from malaria_ssl.detr.build import DetrOutput, expand_queries
from malaria_ssl.detr.loss import DetrCriterion
from malaria_ssl.utils.config import load_experiment

LOSS_CFG = load_experiment("configs/detr/base.yaml")["loss"]


def test_expand_queries_keeps_pretrained_and_tiles():
    old = torch.randn(100, 256)
    new = expand_queries(old, 300, noise=0.1, generator=torch.Generator().manual_seed(0))
    assert new.shape == (300, 256)
    assert torch.equal(new[:100], old)
    # tiled copies + small noise: closest pretrained row of query 100+i is row i
    d = torch.cdist(new[100:], old)
    assert torch.equal(d.argmin(dim=1), torch.arange(200) % 100)
    assert not torch.equal(new[100:200], old)  # noise breaks symmetry


def _perfect_output(targets, num_labels, q=10):
    logits = torch.full((len(targets), q, num_labels + 1), -10.0)
    logits[..., -1] = 10.0  # unmatched queries -> no-object
    boxes = torch.full((len(targets), q, 4), 0.5)
    for b, t in enumerate(targets):
        n = len(t["class_labels"])
        logits[b, :n, :] = -10.0
        logits[b, torch.arange(n), t["class_labels"]] = 10.0
        boxes[b, :n] = t["boxes"]
    return DetrOutput(logits, boxes, [{"logits": logits, "pred_boxes": boxes}] * 5)


def test_criterion_near_zero_for_perfect_predictions_and_aux_terms():
    targets = [{"class_labels": torch.tensor([0, 1]), "boxes": torch.tensor([[0.3, 0.3, 0.1, 0.1], [0.6, 0.6, 0.2, 0.1]])},
               {"class_labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.4, 0.1, 0.2]])}]
    crit = DetrCriterion(2, LOSS_CFG, num_decoder_layers=6, aux_loss=True)
    total, losses = crit(_perfect_output(targets, 2), targets)
    assert total.item() < 1e-3
    assert {"loss_ce", "loss_bbox", "loss_giou", "loss_ce_4", "loss_giou_0"} <= set(losses)


def test_class_weights_change_ce():
    targets = [{"class_labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
    logits = torch.zeros(1, 5, 3)
    logits[..., 2] = 3.0  # every query confidently predicts no-object -> the matched one is wrong
    out = DetrOutput(logits, torch.full((1, 5, 4), 0.5), None)
    base = DetrCriterion(2, LOSS_CFG, 6, aux_loss=False)(out, targets)[1]["loss_ce"]
    weighted = DetrCriterion(2, {**LOSS_CFG, "class_weights": [1.0, 5.0]}, 6, aux_loss=False)
    assert torch.allclose(weighted.image_loss.empty_weight, torch.tensor([1.0, 5.0, LOSS_CFG["eos_coefficient"]]))
    assert weighted(out, targets)[1]["loss_ce"] > base
