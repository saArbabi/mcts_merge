from models.core.train_eval.config_generator import genExpSeires
config = {
 "model_config": {
     "learning_rate": 1e-3,
     "enc_units": 100,
     "dec_units": 100,
     "epochs_n": 50,
     "components_n": 5,
     "batch_size": 256,
    "allowed_error": 0.0,
},
"data_config": {"obs_n": 20,
                "pred_step_n": 7,
                "step_size": 3,
                # "Note": "jerk as target"
                "Note": "lat/long motion not considered jointly"
},
"exp_id": "NA",
"Note": "NA"
}
series_id='series083'
# test_variables = {'pred_step_n':[3, 10]}
# test_variables = {'allowed_error':[[0.7, 0.3], [0.4, 0.15], [0.2, 0.1], [0.1, 0.075]]}
# genExpSeires(series_id=series_id, test_variables=test_variables, config=config)
# genExpSeires(series_id=series_id, test_variable0s=test_variables, config=config)
genExpSeires(series_id=series_id, test_variables=None, config=config)


# ,
# "series040exp004": {
#     "exp_state": "NA",
#     "epoch": 0,
#     "train_loss": "NA",
#     "val_loss": "NA"
# },
# "series040exp005": {
#     "exp_state": "NA",
#     "epoch": 0,
#     "train_loss": "NA",
#     "val_loss": "NA"
# }
