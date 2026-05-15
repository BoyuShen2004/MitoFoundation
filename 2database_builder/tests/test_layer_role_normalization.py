"""Regression tests for layer role normalization.

Key invariants:
- Layers with /labels/inference/ in URL are NEVER shown as ground_truths
- Layers with _pred / _prediction in URL or name are NEVER shown as ground_truths
- Readiness (ready_labeled) requires download_mito_mask_quality = 'good';
  a dataset with only inference/prediction seg layers remains 'EM only'
- True GT layers (no pred cues) keep their ground_truths role
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root or from 2database_builder/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from openorganelle.parser import normalize_layer_role, parse_markdown_text


# ── normalize_layer_role unit tests ───────────────────────────────────────────

class TestNormalizeLayerRole:
    def test_inference_url_overrides_ground_truths(self):
        """The primary regression case: /labels/inference/ path → predictions."""
        role = normalize_layer_role(
            "ground_truths",
            layer_name="mito_seg",
            layer_url=(
                "s3://janelia-cosem-datasets/jrc_mus-liver-7/"
                "jrc_mus-liver-7.n5/labels/inference/segmentations/mito/s0"
            ),
        )
        assert role == "predictions"

    def test_pred_suffix_url_overrides_ground_truths(self):
        """URL with _pred suffix → predictions even if upstream says ground_truths."""
        role = normalize_layer_role(
            "ground_truths",
            layer_name="empanada-mito",
            layer_url="s3://bucket/jrc_mus-liver-7.n5/labels/empanada-mito_pred/s0",
        )
        assert role == "predictions"

    def test_prediction_content_type_overrides_ground_truths(self):
        role = normalize_layer_role(
            "ground_truths",
            layer_name="mito_seg",
            layer_content_type="prediction",
        )
        assert role == "predictions"

    def test_prediction_stage_overrides_ground_truths(self):
        role = normalize_layer_role(
            "ground_truths",
            layer_name="mito_seg",
            layer_stage="prediction",
        )
        assert role == "predictions"

    def test_pred_in_layer_name_overrides_ground_truths(self):
        role = normalize_layer_role(
            "ground_truths",
            layer_name="mito_pred_seg",
        )
        assert role == "predictions"

    def test_genuine_gt_mito_seg_kept(self):
        """True curated GT layer has no inference/pred cues → keep ground_truths."""
        role = normalize_layer_role(
            "ground_truths",
            layer_name="mito_seg",
            layer_url=(
                "s3://janelia-cosem-datasets/jrc_mus-liver-7/"
                "jrc_mus-liver-7.n5/labels/groundtruth/mito_seg/s0"
            ),
            layer_content_type="segmentation",
        )
        assert role == "ground_truths"

    def test_raw_layer_not_affected(self):
        role = normalize_layer_role(
            "raw_or_other_layers",
            layer_name="fibsem-uint8",
            layer_url="s3://bucket/jrc_mus-liver-7.n5/em/fibsem-uint8/s1",
        )
        assert role == "raw_or_other_layers"

    def test_prediction_bucket_stays_predictions(self):
        role = normalize_layer_role(
            "predictions",
            layer_name="empanada-mito",
            layer_url="s3://bucket/labels/empanada-mito/s0",
        )
        assert role == "predictions"

    def test_unknown_upstream_role_falls_back(self):
        role = normalize_layer_role("gt", layer_name="mito_seg")
        assert role == "raw_or_other_layers"

    def test_proofread_name_not_false_positive(self):
        """'proofread' contains no pred cue — should not trigger prediction override."""
        role = normalize_layer_role(
            "ground_truths",
            layer_name="mito_seg_proofread",
            layer_url="s3://bucket/labels/groundtruth/mito_seg_proofread/s0",
        )
        assert role == "ground_truths"


# ── parse_markdown_text end-to-end tests ──────────────────────────────────────

_INFERENCE_ROW = (
    "dataset_name=jrc_mus-liver-7"
    "|ground_truths=image:mito_seg"
    "[content_type=segmentation]"
    "[url=s3://bucket/labels/inference/segmentations/mito/s0]"
)

_GT_ROW = (
    "dataset_name=jrc_mus-liver-7"
    "|ground_truths=image:mito_seg"
    "[content_type=segmentation]"
    "[url=s3://bucket/labels/groundtruth/mito_seg/s0]"
)

_PRED_SUFFIX_ROW = (
    "dataset_name=jrc_mus-liver-7"
    "|ground_truths=image:empanada-mito"
    "[url=s3://bucket/labels/empanada-mito_pred/s0]"
)


class TestParseMarkdownLayerNormalization:
    def _parse_row(self, row: str) -> list:
        md = f"- {row}\n"
        records = parse_markdown_text(md)
        assert len(records) == 1
        return records[0].layers

    def test_inference_url_in_ground_truths_bucket_becomes_predictions(self):
        """Regression: upstream ground_truths + inference URL → predictions role in DB."""
        layers = self._parse_row(_INFERENCE_ROW)
        assert len(layers) == 1
        assert layers[0].layer_role == "predictions", (
            f"Expected 'predictions', got {layers[0].layer_role!r}"
        )

    def test_pred_suffix_url_in_ground_truths_bucket_becomes_predictions(self):
        layers = self._parse_row(_PRED_SUFFIX_ROW)
        assert len(layers) == 1
        assert layers[0].layer_role == "predictions"

    def test_genuine_gt_url_in_ground_truths_bucket_stays(self):
        layers = self._parse_row(_GT_ROW)
        assert len(layers) == 1
        assert layers[0].layer_role == "ground_truths"

    def test_mixed_row_correctly_splits_by_effective_role(self):
        """Row that has both a GT layer and an inference layer in 'ground_truths' bucket."""
        row = (
            "dataset_name=jrc_mus-liver-7"
            "|ground_truths="
            "image:mito_seg[url=s3://bucket/labels/groundtruth/mito_seg/s0]"
            "; image:mito_inference[url=s3://bucket/labels/inference/segmentations/mito/s0]"
        )
        layers = self._parse_row(row)
        assert len(layers) == 2
        roles = {l.layer_name: l.layer_role for l in layers}
        assert roles["mito_seg"] == "ground_truths"
        assert roles["mito_inference"] == "predictions"


# ── classify_layer_gt_pred_raw tests (scraper policy) ─────────────────────────

class TestClassifyLayerGtPredRaw:
    """Tests for mito_layer_policy.classify_layer_gt_pred_raw URL-fix."""

    @pytest.fixture(autouse=True)
    def _import(self):
        global classify_layer_gt_pred_raw
        scraper_root = Path(__file__).resolve().parent.parent.parent / "1web_scraper_01"
        sys.path.insert(0, str(scraper_root))
        from master.deep.mito_layer_policy import classify_layer_gt_pred_raw as fn
        self._fn = fn

    def test_inference_url_classified_as_pred(self):
        cls = self._fn(
            "mito_seg",
            content_type="segmentation",
            url="s3://bucket/labels/inference/segmentations/mito/s0",
        )
        assert cls == "pred", f"Expected 'pred', got {cls!r}"

    def test_pred_suffix_url_classified_as_pred(self):
        cls = self._fn(
            "empanada-mito",
            url="s3://bucket/labels/empanada-mito_pred/s0",
        )
        assert cls == "pred"

    def test_groundtruth_url_classified_as_gt(self):
        cls = self._fn(
            "mito_seg",
            content_type="segmentation",
            url="s3://bucket/labels/groundtruth/mito_seg/s0",
        )
        assert cls == "gt"

    def test_prediction_content_type_classified_as_pred(self):
        cls = self._fn("mito_seg", content_type="prediction")
        assert cls == "pred"

    def test_fibsem_raw_classified_correctly(self):
        cls = self._fn("fibsem-uint8", content_type="em")
        assert cls == "raw"


# ── readiness / quality invariant tests ───────────────────────────────────────

class TestReadinessInvariant:
    """Ensure that inference/prediction layers do not make a dataset 'labeled'."""

    def test_readiness_requires_good_quality_not_inferred_from_role(self):
        """
        Even if a layer is filed under ground_truths in the DB, readiness=labeled
        must come from download_mito_mask_quality='good', not from layer count.

        This is a logic invariant test: the resolved-path code in download_inventory.py
        already uses quality='good' as the guard. We verify that the parser-level
        normalization does NOT incorrectly set mito_mask_quality to 'good'.
        """
        md = (
            "- dataset_name=jrc_mus-liver-7"
            "|ground_truths=image:mito_seg"
            "[content_type=segmentation]"
            "[url=s3://bucket/labels/inference/segmentations/mito/s0]"
            "\n"
        )
        records = parse_markdown_text(md)
        assert len(records) == 1
        ds = records[0]
        # Normalization must not infer quality='good' from a prediction-like layer.
        assert ds.download_mito_mask_quality is None, (
            "Parser must not synthesize mito mask quality from prediction layer"
        )
        # The layer itself should now be predictions.
        assert records[0].layers[0].layer_role == "predictions"
