from models.core.train_eval.experiment_setup import runSeries
import warnings
# test_variables = {'neurons_n':[10,20,30]} # variables being tested
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
runSeries()

# utils.delete_experiment('exp003')
# python run_experiment.py
# tensorboard --logdir experiments
