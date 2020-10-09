from models.core.train_eval.config_generator import genExpSeires
config = {
 "model_config": {
     "learning_rate": 1e-3,
     "enc_units": 20,
     "dec_units": 20,
     "layers_n": 2,
     "epochs_n": 50,
     "batch_n": 128,
     "components_n": 5
},
"data_config": {"step_size": 1,
                "obsSequence_n": 20,
                "pred_horizon": 20,
                "m_s":["vel", "pc"],
                "y_s":["vel", "dx", 'da', 'a_ratio'],
                "Note": "cae setup - with condition: m_df[['vel','pc']], y_df[['vel','dx']]"
},
"exp_id": "NA",
"Note": ""
}
series_id='series002'
genExpSeires(series_id=series_id, test_variables=None, config=config)
