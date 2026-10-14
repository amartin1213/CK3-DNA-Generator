"""Tests for ck3dna.training's PerGeneMSELoss (per-gene-normalized loss),
DiversityPriorLoss (diversity-collapse regularizer), and GenderBCELoss
(binary gender classification loss)."""
import torch

from ck3dna.training import DiversityPriorLoss, GenderBCELoss, PerGeneMSELoss


def test_per_gene_mse_zero_for_identical_tensors():
    lengths = [1, 4, 10]
    loss_fn = PerGeneMSELoss(lengths)
    batch_size = 3
    total = sum(lengths)
    x = torch.rand(batch_size, total)
    assert loss_fn(x, x).item() == 0.0


def test_per_gene_mse_weights_small_and_large_segments_equally():
    """A segment-length-1 gene and a segment-length-100 gene should
    contribute equally to the loss, unlike plain element-wise MSE where the
    100-dim gene would dominate by sheer element count.
    """
    lengths = [1, 100]
    loss_fn = PerGeneMSELoss(lengths)

    # Perfect prediction on the big segment, off-by-10 on the tiny one.
    target = torch.zeros(1, 101)
    pred = torch.zeros(1, 101)
    pred[0, 0] = 10.0  # error only in the length-1 segment

    per_gene = loss_fn(pred, target).item()
    plain = torch.nn.functional.mse_loss(pred, target).item()

    # Plain MSE dilutes the length-1 gene's error across all 101 elements;
    # per-gene MSE should report a much larger loss for the same error,
    # since that gene's full weight is 1/2 (not 1/101) of the total.
    assert per_gene > plain
    # The length-1 gene alone has MSE 100 (10**2); averaged over 2 genes
    # (the length-100 gene contributes 0) that's 50.
    assert per_gene == 50.0


def test_per_gene_mse_matches_elementwise_when_segments_equal_length():
    lengths = [5, 5, 5]
    loss_fn = PerGeneMSELoss(lengths)
    target = torch.rand(2, 15)
    pred = torch.rand(2, 15)
    per_gene = loss_fn(pred, target).item()
    plain = torch.nn.functional.mse_loss(pred, target).item()
    assert abs(per_gene - plain) < 1e-5


def test_per_gene_mse_group_weights_upweight_matching_genes():
    """`group_weights` should give genes matching a name substring a larger
    share of the averaged loss, without needing a weight per individual
    gene."""
    lengths = [1, 1, 1]
    names = ["gene_mouth_size", "gene_jaw_def", "gene_eye_color"]
    loss_fn = PerGeneMSELoss(lengths, names=names, group_weights={"mouth": 3.0, "jaw": 3.0})

    target = torch.zeros(1, 3)
    pred = torch.zeros(1, 3)
    pred[0, 0] = 10.0  # error only in the "mouth" gene

    weighted = loss_fn(pred, target).item()
    # Equal weighting (facial_structure_weight disabled) would give
    # 100/3 ~= 33.33; with 3x weight on mouth/jaw, mouth's full-error
    # contributes 3x its share: (3*100)/(3+3+1) = 42.857...
    assert weighted > 100.0 / 3.0
    assert abs(weighted - (3 * 100.0) / 7.0) < 1e-4

    # Flip the error onto the unweighted "eye_color" gene instead: it should
    # contribute *less* than before since the weighted denominator is larger.
    pred2 = torch.zeros(1, 3)
    pred2[0, 2] = 10.0
    unweighted_gene_loss = loss_fn(pred2, target).item()
    assert unweighted_gene_loss < weighted
    assert abs(unweighted_gene_loss - 100.0 / 7.0) < 1e-4


def test_diversity_prior_loss_zero_when_batch_already_diverse():
    lengths = [1, 4]
    loss_fn = DiversityPriorLoss(lengths, target_stds=[1.0, 2.0])
    # Hand-crafted batch where every segment already has std well above target.
    pred = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [100.0, 50.0, 50.0, 50.0, 50.0],
            [-100.0, -50.0, -50.0, -50.0, -50.0],
        ]
    )
    assert loss_fn(pred).item() == 0.0


def test_diversity_prior_loss_positive_when_batch_collapsed():
    lengths = [1]
    loss_fn = DiversityPriorLoss(lengths, target_stds=[10.0])
    # Every row identical -> batch std is 0, fully collapsed.
    pred = torch.zeros(4, 1)
    penalty = loss_fn(pred).item()
    assert penalty == 100.0  # relu(10 - 0)**2


def test_diversity_prior_loss_handles_batch_size_one():
    lengths = [1, 4]
    loss_fn = DiversityPriorLoss(lengths, target_stds=[1.0, 2.0])
    pred = torch.rand(1, 5)
    # Std across a single sample is undefined; should degrade gracefully to 0
    # rather than NaN/crash.
    assert loss_fn(pred).item() == 0.0


def test_gender_bce_loss_low_when_confident_and_correct():
    loss_fn = GenderBCELoss()
    # Confident, correct predictions for both genders.
    pred = torch.zeros(2, 5)
    pred[0, 0] = 254.0  # predicted boy
    pred[1, 0] = 1.0  # predicted girl
    target = torch.zeros(2, 5)
    target[0, 0] = 255.0  # true boy
    target[1, 0] = 0.0  # true girl
    assert loss_fn(pred, target).item() < 0.05


def test_gender_bce_loss_high_when_confidently_wrong():
    loss_fn = GenderBCELoss()
    pred = torch.zeros(1, 5)
    pred[0, 0] = 254.0  # confidently predicted boy
    target = torch.zeros(1, 5)
    target[0, 0] = 0.0  # true girl -- confidently wrong
    assert loss_fn(pred, target).item() > 3.0


def test_gender_bce_loss_higher_for_confident_wrong_than_hedged():
    """A model hedging near the midpoint for an ambiguous photo should be
    penalized less than one confidently predicting the wrong gender, but
    more than one confidently predicting the right one -- BCE's asymmetric
    penalty is what should discourage the midpoint hedge that plain MSE
    tolerates."""
    loss_fn = GenderBCELoss()
    target = torch.zeros(1, 5)
    target[0, 0] = 255.0  # true boy

    hedged = torch.zeros(1, 5)
    hedged[0, 0] = 127.5
    wrong = torch.zeros(1, 5)
    wrong[0, 0] = 1.0
    correct = torch.zeros(1, 5)
    correct[0, 0] = 254.0

    hedged_loss = loss_fn(hedged, target).item()
    wrong_loss = loss_fn(wrong, target).item()
    correct_loss = loss_fn(correct, target).item()
    assert correct_loss < hedged_loss < wrong_loss
