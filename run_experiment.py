from models.core.train_eval.experiment_setup import runSeries
from models.core.train_eval.config_generator import genExpSeires
import warnings
test_variables = {'neurons_n':[10,20,30]} # variables being tested

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
genExpSeires(config_base='baseline_test.json', test_variables=test_variables)
runSeries()

# utils.delete_experiment('exp003')
# python run_experiment.py
# tensorboard --logdir experiments
