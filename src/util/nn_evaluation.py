#!{sys.executable} -m pip install pytorch-ignite # helped to fix import issue with ignite
from ignite.engine import *


def eval_step(engine, batch):
    return batch


def get_default_evaluator():
    default_evaluator = Engine(eval_step)
    return default_evaluator


def get_test_evaluator():
    test_evaluator = Engine(eval_step)
    return test_evaluator
