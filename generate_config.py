from models.core.train_eval.config_generator import genExpSeires
config = {
 "model_config": {
     "learning_rate": 1e-2,
     "neurons_n": 50,
     "layers_n": 2,
     "epochs_n": 5,
     "batch_n": 128,
     "components_n": 5
},
"data_config": {"step_size": 3,
                "obsSequence_n": 1,
                "m_s":["vel", "pc", "act_long_p"],
                "y_s":["vel", "dv", "dx", "da", "a_ratio"],
                "retain":["vel"],
},
"exp_id": "NA",
"model_type": "merge_policy",
"Note": "NA"
}

genExpSeires(config=config, test_variables=None)
config['data_config']['obsSequence_n']
