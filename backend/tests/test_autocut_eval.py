"""tools/autocut_eval.py's scoring math, against synthetic spans -- no video,
no transcript, no model. See AUTO_EDIT_IMPROVEMENT_PLAN.md §2, §6 for what the
auto-edit's `reason` vocabulary and the "re-run N times" rationale are.
"""

from tools import autocut_eval as ev


def _span(start, end, kind=None):
    d = {"start": start, "end": end}
    if kind is not None:
        d["kind"] = kind
    return d


# --- iou ---------------------------------------------------------------

def test_iou_of_identical_spans_is_1():
    a = _span(10.0, 20.0)
    assert ev.iou(a, dict(a)) == 1.0


def test_iou_of_disjoint_spans_is_0():
    assert ev.iou(_span(0.0, 5.0), _span(10.0, 15.0)) == 0.0


def test_iou_of_a_half_overlap():
    # [0,10] and [5,15]: intersection 5, union 15 -> 1/3
    assert round(ev.iou(_span(0.0, 10.0), _span(5.0, 15.0)), 4) == round(1 / 3, 4)


# --- matching ------------------------------------------------------------

def test_match_spans_iou_greedy_prefers_the_best_overlap():
    predicted = [_span(0.0, 10.0), _span(9.0, 20.0)]
    ground_truth = [_span(0.0, 10.0)]
    matches, unmatched_pred, unmatched_gt = ev.match_spans_iou(predicted, ground_truth, 0.5)
    assert len(matches) == 1
    assert matches[0]["predicted_index"] == 0
    assert unmatched_pred == [1]
    assert unmatched_gt == []


def test_match_spans_iou_respects_threshold():
    # Overlap 1/19 << 0.5
    predicted = [_span(0.0, 10.0)]
    ground_truth = [_span(9.0, 19.0)]
    matches, unmatched_pred, unmatched_gt = ev.match_spans_iou(predicted, ground_truth, 0.5)
    assert matches == []
    assert unmatched_pred == [0]
    assert unmatched_gt == [0]


# --- span precision/recall ------------------------------------------------

def test_precision_recall_perfect_match():
    spans = [_span(0.0, 2.0), _span(5.0, 8.0)]
    stats = ev.precision_recall_by_span(spans, [dict(s) for s in spans])
    assert stats["precision"] == 1.0
    assert stats["recall"] == 1.0
    assert stats["f1"] == 1.0
    assert stats["matched"] == 2


def test_precision_recall_with_a_false_positive_and_false_negative():
    predicted = [_span(0.0, 2.0), _span(50.0, 52.0)]     # second is a false positive
    ground_truth = [_span(0.0, 2.0), _span(10.0, 12.0)]  # second is missed (false negative)
    stats = ev.precision_recall_by_span(predicted, ground_truth)
    assert stats["matched"] == 1
    assert stats["precision"] == 0.5
    assert stats["recall"] == 0.5


def test_precision_recall_both_empty_is_vacuously_perfect():
    stats = ev.precision_recall_by_span([], [])
    assert stats["precision"] == 1.0
    assert stats["recall"] == 1.0


# --- seconds-based stats --------------------------------------------------

def test_seconds_overlap_stats_perfect_overlap():
    spans = [_span(0.0, 4.0), _span(10.0, 12.0)]
    stats = ev.seconds_overlap_stats(spans, [dict(s) for s in spans])
    assert stats["predicted_total_s"] == 6.0
    assert stats["ground_truth_total_s"] == 6.0
    assert stats["overlap_s"] == 6.0
    assert stats["precision_s"] == 1.0
    assert stats["recall_s"] == 1.0
    assert stats["content_wrongly_removed_s"] == 0.0
    assert stats["missed_debris_s"] == 0.0


def test_seconds_overlap_stats_content_wrongly_removed_and_missed_debris():
    # Predicted cuts [0,5] and [20,22]; ground truth only wants [0,3] and [8,10].
    predicted = [_span(0.0, 5.0), _span(20.0, 22.0)]
    ground_truth = [_span(0.0, 3.0), _span(8.0, 10.0)]
    stats = ev.seconds_overlap_stats(predicted, ground_truth)
    # overlap is just [0,3] = 3s
    assert stats["overlap_s"] == 3.0
    # predicted total = 7s, so 7 - 3 = 4s cut that was not in any GT span
    assert stats["content_wrongly_removed_s"] == 4.0
    # ground truth total = 5s, so 5 - 3 = 2s of debris the predicted cuts missed
    assert stats["missed_debris_s"] == 2.0


def test_seconds_overlap_stats_merges_overlapping_spans_within_a_list():
    # Two overlapping predicted spans should not double-count their overlap.
    predicted = [_span(0.0, 5.0), _span(3.0, 8.0)]
    ground_truth = [_span(0.0, 8.0)]
    stats = ev.seconds_overlap_stats(predicted, ground_truth)
    assert stats["predicted_total_s"] == 8.0
    assert stats["overlap_s"] == 8.0


def test_seconds_overlap_stats_empty_predicted_or_ground_truth():
    stats = ev.seconds_overlap_stats([], [_span(0.0, 5.0)])
    assert stats["precision_s"] == 1.0  # vacuous: no predicted seconds to be wrong about
    assert stats["recall_s"] == 0.0
    assert stats["missed_debris_s"] == 5.0


# --- per-kind recall -------------------------------------------------------

def test_per_kind_recall_counts_each_kind_independently():
    predicted = [_span(0.0, 2.0, "fumble"), _span(10.0, 12.0, "retake")]
    ground_truth = [
        _span(0.0, 2.0, "fumble"),
        _span(10.0, 12.0, "retake"),
        _span(20.0, 22.0, "false_start"),  # not found by any predicted span
    ]
    result = ev.per_kind_recall(predicted, ground_truth)
    assert result["fumble"]["recall"] == 1.0
    assert result["retake"]["recall"] == 1.0
    assert result["false_start"]["recall"] == 0.0
    assert result["false_start"]["total"] == 1


def test_per_kind_recall_a_kind_with_no_ground_truth_spans_is_absent():
    predicted = []
    ground_truth = [_span(0.0, 2.0, "silence")]
    result = ev.per_kind_recall(predicted, ground_truth)
    assert set(result.keys()) == {"silence"}
    assert result["silence"]["recall"] == 0.0


# --- score_run / aggregate_runs -------------------------------------------

def test_score_run_bundles_every_metric():
    predicted = [_span(0.0, 2.0, "fumble")]
    ground_truth = [_span(0.0, 2.0, "fumble")]
    result = ev.score_run(predicted, ground_truth)
    assert "span" in result and "seconds" in result and "per_kind" in result
    assert result["span"]["recall"] == 1.0


def test_aggregate_runs_averages_across_runs():
    ground_truth = [_span(0.0, 2.0, "fumble")]
    run_a = ev.score_run([_span(0.0, 2.0, "fumble")], ground_truth)   # perfect
    run_b = ev.score_run([], ground_truth)                            # misses everything
    agg = ev.aggregate_runs([run_a, run_b])
    assert agg["runs"] == 2
    assert agg["span"]["recall"]["mean"] == 0.5
    assert agg["per_kind"]["fumble"]["mean_recall"] == 0.5


def test_aggregate_runs_of_nothing_is_zero_runs():
    assert ev.aggregate_runs([]) == {"runs": 0}


# --- cut_spans_from_words --------------------------------------------------

def _word(start_s, end_s, enabled=True, reason=None, fps=30):
    return {"start_frame": round(start_s * fps), "end_frame": round(end_s * fps),
           "enabled": enabled, "reason": reason}


def test_cut_spans_from_words_tags_a_disabled_run_by_its_reason():
    words = [
        _word(0.0, 1.0, enabled=True),
        _word(1.0, 1.5, enabled=False, reason="filler_sound"),
        _word(1.5, 2.0, enabled=False, reason="filler_sound"),
        _word(2.0, 3.0, enabled=True),
    ]
    spans = ev.cut_spans_from_words(words, fps_num=30, fps_den=1)
    assert len(spans) == 1
    assert spans[0]["kind"] == "fumble"
    assert spans[0]["start"] == 1.0
    assert spans[0]["end"] == 2.0


def test_cut_spans_from_words_maps_retake_and_false_start():
    words = [
        _word(0.0, 1.0, enabled=True),
        _word(1.0, 2.0, enabled=False, reason="retake"),
        _word(2.0, 3.0, enabled=True),
        _word(3.0, 4.0, enabled=False, reason="false_start"),
        _word(4.0, 5.0, enabled=True),
    ]
    spans = ev.cut_spans_from_words(words, fps_num=30, fps_den=1)
    kinds = {round(s["start"]): s["kind"] for s in spans}
    assert kinds[1] == "retake"
    assert kinds[3] == "false_start"


def test_cut_spans_from_words_flags_a_wide_gap_between_enabled_words_as_silence():
    words = [
        _word(0.0, 1.0, enabled=True),
        _word(5.0, 6.0, enabled=True),   # 4s gap, nothing planned in between
    ]
    spans = ev.cut_spans_from_words(words, fps_num=30, fps_den=1, silence_gap_floor_s=0.3)
    assert len(spans) == 1
    assert spans[0]["kind"] == "silence"
    assert spans[0]["start"] == 1.0
    assert spans[0]["end"] == 5.0


def test_cut_spans_from_words_ignores_a_small_gap_below_the_floor():
    words = [
        _word(0.0, 1.0, enabled=True),
        _word(1.1, 2.0, enabled=True),   # 0.1s gap, ordinary word spacing
    ]
    spans = ev.cut_spans_from_words(words, fps_num=30, fps_den=1, silence_gap_floor_s=0.3)
    assert spans == []


def test_cut_spans_from_words_accepts_word_objects_with_model_dump():
    class _FakeWord:
        def __init__(self, **kw):
            self._kw = kw

        def model_dump(self):
            return dict(self._kw)

    words = [
        _FakeWord(start_frame=0, end_frame=30, enabled=True, reason=None),
        _FakeWord(start_frame=30, end_frame=60, enabled=False, reason="stutter"),
        _FakeWord(start_frame=60, end_frame=90, enabled=True, reason=None),
    ]
    spans = ev.cut_spans_from_words(words, fps_num=30, fps_den=1)
    assert len(spans) == 1
    assert spans[0]["kind"] == "fumble"
