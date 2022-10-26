import pytest 
import yaml
import sys

sys.path.append('/projects/CPINO')
from train import setup_problem, main

class config_test: 
    def __init__(self, config, problem=None) -> None:
        self.config = config
        self.problem = problem

        

@pytest.fixture
def setup_fixture(): 
    with open('../experiments/base_configs/NS3D.yaml', 'r') as stream:
        NSconfig = yaml.load(stream, yaml.FullLoader)
    problem = setup_problem(NSconfig)
    return config_test(NSconfig, problem)

@pytest.mark.parametrize("fixture,model", [('setup_fixture',"PINO"), ('setup_fixture',"CPINO"), ('setup_fixture',"SAPINO")])
def test_model(fixture, model, request): 
    config_test = request.getfixturevalue(fixture)
    config = config_test.config
    problem = config_test.problem
    config["info"]["model"] = model
    with pytest.raises(SystemExit):
        main(config, problem, log=False)
    

