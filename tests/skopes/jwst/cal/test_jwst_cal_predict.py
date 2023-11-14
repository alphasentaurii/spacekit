from pytest import mark
from spacekit.skopes.jwst.cal.predict import JwstCalPredict, predict_handler


EXPECTED = {
    "niriss": {"gbSize": 2.57},
    "miri": {"gbSize": 0.79},
    "nircam": {"gbSize": 3.8},
}


@mark.jwst
@mark.predict
def test_jwst_cal_predict(jwstcal_input_path):
    jcal = JwstCalPredict(input_path=jwstcal_input_path)
    assert jcal.img3_reg.__name__ == "Builder"
    assert jcal.img3_reg.blueprint == "jwst_img3_reg"
    assert jcal.img3_reg.model_path == 'models/jwst_cal/img3_reg/img3_reg.keras'
    assert jcal.tx_file == 'models/jwst_cal/img3_reg/tx_data.json'
    assert jcal.img3_reg.model.name == 'img3_reg'
    assert len(jcal.img3_reg.model.layers) == 10
    jcal.run_inference()
    assert len(jcal.input_data['IMAGE']) == 3
    assert jcal.inputs['IMAGE'].shape == (3, 18)
    for k, v in jcal.predictions.items():
        instr = k.split("_")[1]
        assert EXPECTED[instr]["gbSize"] == v["gbSize"]


@mark.jwst
@mark.predict
def test_jwst_cal_predict_handler(jwstcal_input_path):
    jcal = predict_handler(jwstcal_input_path)
    assert len(jcal.predictions) == 3
    for k, v in jcal.predictions.items():
        instr = k.split("_")[1]
        assert EXPECTED[instr]["gbSize"] == v["gbSize"]

#TODO
@mark.jwst
@mark.predict
def test_jwst_cal_predict_input_single_file_path(jwstcal_file_path):
    # jwstcal_file_path = "jw02732005001_02105_00001_mirimage_uncal.fits"
    jcal = JwstCalPredict(input_path=jwstcal_file_path)
    # should find only two files:
    # "jw02732005001_02105_00001_mirimage_uncal.fits"
    # "jw02732005001_02105_00002_mirimage_uncal.fits"

#TODO
@mark.jwst
@mark.predict
def test_jwst_cal_predict_pid_obs(jwstcal_input_path):
    # jwstcal_file_path = "jw02732005001_02105_0001_mirimage_uncal.fits"
    jcal1 = JwstCalPredict(input_path=jwstcal_input_path, pid=2732, obs=1)
    jcal2 = JwstCalPredict(input_path=jwstcal_input_path, pid=2732, obs="005")
    jcal3 = JwstCalPredict(input_path=jwstcal_input_path, pid=2732, obs="5")
