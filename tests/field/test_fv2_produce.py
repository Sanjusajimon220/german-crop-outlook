import fv2_produce as P
import phase11_fusion_test as F


def test_hide_share_zero_hides_nothing():
    S = F.Season("pilot", 2024, "lai", landsat=False, date_hide=True, hide_share=0.0)
    assert not S.hide.any()


def test_uncertainty_table_keys_and_order():
    u = P.uncertainty_table()
    for t in ("lai", "ndvi"):
        assert set(u[t]) == {"line_le15", "line_16_30", "radar_gt30", "live_le30", "live_gt30"}
        assert all(0 < v < (2.0 if t == "lai" else 0.5) for v in u[t].values())


def test_design_adopted_reads_result_lines(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("x\nRESULT lai: V2 beats ... -> DESIGN ADOPTED\nRESULT ndvi: V2 beats ... -> DESIGN NOT adopted\n")
    assert P.design_adopted("lai", log) is True
    assert P.design_adopted("ndvi", log) is False


def test_ndvi_measured_sd_constant():
    assert P.NDVI_MEASURED_SD == 0.02
