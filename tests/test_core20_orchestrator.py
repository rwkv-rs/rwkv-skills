from ops.g1i_strict46.continue_core20_campaign import LANES, _model_state


def test_core20_lanes_use_only_reserved_inference_resources():
    lane_157 = LANES["157"]
    lane_8222 = LANES["8222"]

    assert (lane_157.gpu, lane_157.port) == (3, 19439)
    assert (lane_8222.gpu, lane_8222.port) == (2, 18074)
    assert all("18073" not in model.service for model in lane_8222.models)
    assert all("g1i-13.3b" not in model.service for model in lane_8222.models)


def test_model_state_counts_only_declared_states():
    audit = {
        "cells": [
            {"model": "m", "state": "valid"},
            {"model": "m", "state": "running"},
            {"model": "m", "state": "missing"},
            {"model": "m", "state": "invalid"},
        ]
    }
    assert _model_state({"cells": audit["cells"] + [
        {"model": "m", "state": "valid"} for _ in range(30)
    ]}, "m") == {
        "valid": 31,
        "running": 1,
        "missing": 1,
        "invalid": 1,
    }
