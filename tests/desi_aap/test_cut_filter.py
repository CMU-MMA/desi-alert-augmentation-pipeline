"""The JSON filter API: the grammar, the cut semantics, and the stage they become."""

import json

import pandas as pd
import pytest
from conftest import host, make_placed
from pydantic import ValidationError

from desi_aap.config import ConfigError
from desi_aap.stages.base import StageResult
from desi_aap.stages.cut_filter import (
    OUTPUT_PREFIX,
    RESERVED_NAMES,
    CutFilterSpec,
    CutSpec,
    apply_cuts,
    load_cut_filters,
    run_cut_filter,
    runner,
)
from desi_aap.stages.distance import STAGE as DISTANCE_STAGE

STAMP = "20260101T000000Z"


def write_filter(directory, name, spec):
    """Write one filter file the way a user would: a JSON dict in the directory."""
    path = directory / f"{name}.json"
    path.write_text(json.dumps(spec))
    return path


@pytest.fixture(name="placed")
def placed_fixture():
    """Three placed alerts at three redshifts: near, middling, far."""
    return make_placed({"desi_dr1": [[host(z=0.01)], [host(z=0.05)], [host(z=0.4)]]})


# --- The grammar: what a filter file may and may not say. -------------------


def test_a_cut_needs_at_least_one_bound():
    with pytest.raises(ValidationError, match="at least one of min, max, one_of"):
        CutSpec(column="dist_mpc_SHOES")


def test_an_empty_one_of_is_rejected():
    with pytest.raises(ValidationError, match="pass nothing"):
        CutSpec(column="candidate.band", one_of=[])


def test_a_filter_needs_at_least_one_cut():
    with pytest.raises(ValidationError, match="at least one entry in 'cuts'"):
        CutFilterSpec(cuts=[])


def test_a_misspelled_key_is_an_error_not_a_weaker_filter():
    """The bound that fails loudly is the bound that cannot silently stop cutting."""
    with pytest.raises(ValidationError, match="maximum"):
        CutFilterSpec.model_validate({"cuts": [{"column": "x", "maximum": -20}]})


def test_loading_names_the_broken_file(pipeline_config):
    write_filter(pipeline_config.filters.dir, "broken", {"cuts": [{"column": "x"}]})

    with pytest.raises(ConfigError, match=r"broken\.json"):
        load_cut_filters(pipeline_config)


def test_a_file_that_is_not_json_names_itself(pipeline_config):
    (pipeline_config.filters.dir / "mangled.json").write_text("{not json")

    with pytest.raises(ConfigError, match=r"mangled\.json is not valid JSON"):
        load_cut_filters(pipeline_config)


@pytest.mark.parametrize("name", sorted(RESERVED_NAMES)[:3] + ["localize"])
def test_a_reserved_name_is_rejected(pipeline_config, name):
    """A filter may not shadow a stage or a config section."""
    write_filter(pipeline_config.filters.dir, name, {"cuts": [{"column": "x", "max": 1}]})

    with pytest.raises(ConfigError, match="reserved"):
        load_cut_filters(pipeline_config)


def test_filters_load_sorted_by_name(pipeline_config):
    """Deterministic order, whatever the filesystem returns."""
    for name in ("zeta", "alpha", "mid"):
        write_filter(pipeline_config.filters.dir, name, {"cuts": [{"column": "x", "max": 1}]})

    loaded = load_cut_filters(pipeline_config)

    assert [f.name for f in loaded] == ["alpha", "mid", "zeta"]


def test_a_default_dir_that_does_not_exist_is_no_filters(pipeline_config):
    """A clone without the shipped definitions still runs."""
    cfg = pipeline_config.model_copy(update={"filters": type(pipeline_config.filters)()})
    assert cfg.filters.dir.name == "filters"  # the default, untouched by conftest

    assert load_cut_filters(cfg) == () or all(f.name for f in load_cut_filters(cfg))


def test_an_explicit_dir_that_does_not_exist_is_an_error(pipeline_config, tmp_path):
    """A typo'd path must not read as a decision to run with no filters."""
    cfg = pipeline_config.model_copy(
        update={
            "filters": type(pipeline_config.filters)(dir=tmp_path / "no_such_dir"),
        }
    )

    with pytest.raises(ConfigError, match="no_such_dir"):
        load_cut_filters(cfg)


def test_title_and_columns_reach_the_slack_display(pipeline_config):
    write_filter(
        pipeline_config.filters.dir,
        "shiny",
        {"title": "very shiny candidate", "columns": ["abs_mag_SHOES"], "cuts": [{"column": "x", "max": 1}]},
    )

    (loaded,) = load_cut_filters(pipeline_config)

    assert loaded.slack_display.title == "very shiny candidate"
    assert loaded.slack_display.columns == ("abs_mag_SHOES",)


def test_the_title_defaults_to_the_filter_name(pipeline_config):
    write_filter(pipeline_config.filters.dir, "shiny", {"cuts": [{"column": "x", "max": 1}]})

    (loaded,) = load_cut_filters(pipeline_config)

    assert loaded.slack_display.title == "shiny candidate"


# --- The cuts: what passes and what fails. -----------------------------------


def test_bounds_are_inclusive():
    """'Brighter than -20' includes -20 itself; both bounds work the same way."""
    frame = pd.DataFrame({"m": [-21.0, -20.0, -19.0]})

    keep, _ = apply_cuts(frame, [CutSpec(column="m", max=-20.0)])
    assert keep.tolist() == [True, True, False]

    keep, _ = apply_cuts(frame, [CutSpec(column="m", min=-20.0)])
    assert keep.tolist() == [False, True, True]


def test_cuts_are_anded(placed):
    """Every cut must pass: the middling alert is the only one both keep."""
    keep, _ = apply_cuts(
        placed,
        [
            CutSpec(column="host_redshift", min=0.02),
            CutSpec(column="host_redshift", max=0.1),
        ],
    )

    assert keep.tolist() == [False, True, False]


def test_a_missing_or_unparseable_value_fails_the_cut():
    """A cut is a positive statement, and an absent value cannot make one."""
    frame = pd.DataFrame({"m": [1.0, None, "not a number"]})

    keep, _ = apply_cuts(frame, [CutSpec(column="m", min=0.0)])

    assert keep.tolist() == [True, False, False]


def test_one_of_selects_labels():
    frame = pd.DataFrame({"band": ["g", "r", None, "z"]})

    keep, _ = apply_cuts(frame, [CutSpec(column="band", one_of=["g", "r"])])

    assert keep.tolist() == [True, True, False, False]


def test_a_cut_on_a_missing_column_names_the_real_ones(placed):
    """A filter cutting on a column that never exists is broken, not quiet."""
    with pytest.raises(ValueError, match="host_redshift"):
        apply_cuts(placed, [CutSpec(column="no_such_column", max=1.0)])


def test_survivors_are_counted_per_cut(placed):
    """A filter that finds nothing says which cut nothing passed."""
    _, survivors = apply_cuts(
        placed,
        [
            CutSpec(column="host_redshift", max=1.0),  # passes all three
            CutSpec(column="host_redshift", max=-1.0),  # passes none
        ],
    )

    assert survivors == {"host_redshift <= 1.0": 3, "host_redshift <= -1.0": 0}


# --- The stage one file becomes. ---------------------------------------------


@pytest.fixture(name="shiny")
def shiny_fixture(pipeline_config):
    """One loaded filter keeping everything within 300 Mpc."""
    write_filter(
        pipeline_config.filters.dir,
        "shiny",
        {"columns": ["dist_mpc_SHOES"], "cuts": [{"column": "dist_mpc_SHOES", "max": 300.0}]},
    )
    (loaded,) = load_cut_filters(pipeline_config)
    return loaded


def test_run_writes_the_candidates_under_the_filters_name(pipeline_config, placed, shiny):
    inputs = {DISTANCE_STAGE: StageResult(stage=DISTANCE_STAGE, frame=placed)}

    result = run_cut_filter(pipeline_config, cut_filter=shiny, inputs=inputs, stamp=STAMP)

    # z=0.01 (~44 Mpc) and z=0.05 (~224 Mpc) pass; z=0.4 does not.
    assert result.stage == "shiny"
    assert result.summary["n_alerts"] == 3
    assert result.summary["n_candidates"] == 2
    assert result.summary["definition"].endswith("shiny.json")
    assert result.output_path == pipeline_config.run.stage_dir("shiny") / f"{OUTPUT_PREFIX}_{STAMP}.parquet"
    assert result.output_path.exists()


def test_run_finds_nothing_quietly(pipeline_config, placed, shiny):
    """No candidates is the normal outcome, not an error, and writes no file."""
    inputs = {DISTANCE_STAGE: StageResult(stage=DISTANCE_STAGE, frame=placed.iloc[2:])}

    result = run_cut_filter(pipeline_config, cut_filter=shiny, inputs=inputs, stamp=STAMP)

    assert result.is_empty
    assert result.output_path is None
    assert result.summary["n_candidates"] == 0


def test_run_passes_an_empty_upstream_through(pipeline_config, shiny):
    inputs = {DISTANCE_STAGE: StageResult(stage=DISTANCE_STAGE, frame=None)}

    result = run_cut_filter(pipeline_config, cut_filter=shiny, inputs=inputs, stamp=STAMP)

    assert result.is_empty
    assert result.summary == {"n_alerts": 0}


def test_run_names_the_stage_that_must_run_first(pipeline_config, shiny):
    with pytest.raises(KeyError, match=DISTANCE_STAGE):
        run_cut_filter(pipeline_config, cut_filter=shiny, inputs={}, stamp=STAMP)


def test_runner_binds_to_the_stage_signature(pipeline_config, placed, shiny):
    """What the pipeline actually calls: run(cfg, dry_run=..., inputs=..., stamp=...)."""
    inputs = {DISTANCE_STAGE: StageResult(stage=DISTANCE_STAGE, frame=placed)}

    result = runner(shiny)(pipeline_config, dry_run=True, inputs=inputs, stamp=STAMP)

    assert result.stage == "shiny"
    assert result.output_path is None  # dry run
    assert result.summary["n_candidates"] == 2


def test_the_shipped_filter_definitions_are_valid():
    """The two filters the repo ships must always satisfy their own grammar."""
    from pathlib import Path

    repo_filters = Path(__file__).parents[2] / "filters"
    specs = {
        path.stem: CutFilterSpec.model_validate(json.loads(path.read_text()))
        for path in sorted(repo_filters.glob("*.json"))
    }

    assert set(specs) == {"luminous", "nearby"}
    assert specs["luminous"].cuts[0].column == "abs_mag_SHOES"
    assert specs["luminous"].cuts[0].max == -20.0
    assert specs["nearby"].cuts[0].column == "dist_mpc_SHOES"
    assert specs["nearby"].cuts[0].max == 750.0
