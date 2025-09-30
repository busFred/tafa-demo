from __future__ import annotations

from typing import Optional

import lightning as pl
import mymodels
import tensordict as thd
import torch as th
import torch.distributions.utils
import tqdm.auto as tqdm

from . import makers


@th.no_grad()
def identify_init_fidx(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    max_features: Optional[int],
    n_repeat: int,
    n_masks: int,
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
) -> tuple[int, th.Tensor]:
    """a function to find initial feature index

    Args:
        tdata (thd.TensorDict): (n, ) training data and must contain `xs` and `ys`
        classifier (mymodels.classifiers.SubsetFeatureClassifier): a subset feature classifier
        max_features (Optional[int]): optional maximum number of features; if `none`, no restriction on number of features included in template.
        n_repeat (int): number of trials to search for initial feature
        n_masks (int): number of candidates to condsider for each trail
        lmbda (float): penalty term for choosing more features
        bsz (int): batch size

    Returns::
        int: initial feature index
        th.Tensor: (n_covs, ) the costs of each feature
    """
    best_fms: th.Tensor = th.stack(
        [
            _ident_init_fidx_single(
                tdata=tdata,
                classifier=classifier,
                max_features=max_features,
                n_masks=n_masks,
                lmbda=lmbda,
                bsz=bsz,
                plf=plf,
            )[1]
            for _ in tqdm.trange(
                n_repeat, desc="ident init_fidx", leave=False, dynamic_ncols=True
            )
        ]
    )
    best_fm: th.Tensor = th.mean(best_fms, dim=0)
    return int(th.argmin(best_fm)), best_fm


@th.no_grad()
def _ident_init_fidx_single(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    max_features: Optional[int],
    n_masks: int,
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
    generator: Optional[th.Generator] = None,
) -> tuple[int, th.Tensor]:
    generator = th.default_generator if generator is None else generator
    n_covs: int = tdata["xs"].shape[1]
    classifier.eval().to(device=plf.device)
    if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
        classifier.fit_(
            makers.candidates.make_feature_masks(
                n_covs=n_covs,
                n_masks=len(tdata),
                min_features=1,
                max_features=max_features,
                generator=generator,
            )
        )
    fms: th.Tensor = makers.candidates.make_feature_masks(
        n_covs=n_covs,
        n_masks=n_masks,
        min_features=1,
        max_features=max_features,
        generator=generator,
    )
    best_mask_cost: th.Tensor = th.inf * th.ones((n_covs,))
    n_masks = min(n_masks, len(fms))
    for _i in tqdm.trange(
        n_masks, desc="ident_init_fidx_single", leave=False, dynamic_ncols=True
    ):
        # (n_covs, )
        _bfm: th.Tensor = fms[_i]
        # (bsz, )
        _btidxs: th.Tensor = th.randint(0, len(tdata), (bsz,), generator=generator)
        # (bsz, n_covs)
        _btdata: thd.TensorDict = tdata[_btidxs]
        _bfms: th.Tensor = _bfm[None, :].expand(bsz, -1)
        # (bsz, n_labels)
        _bpyhats: th.Tensor = classifier.predict_proba(
            _btdata["xs"].to(device=plf.device), _bfms.to(device=plf.device)
        ).to(device="cpu")
        _blyhats: th.Tensor = torch.distributions.utils.probs_to_logits(_bpyhats)
        # ()
        _bcosts: th.Tensor = th.nn.functional.cross_entropy(
            _blyhats, _btdata["ys"]
        ) + lmbda * th.sum(_bfm)
        best_mask_cost = th.minimum(
            best_mask_cost, th.where(_bfm == 1, _bcosts, th.inf)
        )
    return int(th.argmin(best_mask_cost).item()), best_mask_cost
