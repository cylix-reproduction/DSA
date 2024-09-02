from config.attack_config import cfg

from .i_fgsm import FGSM, IFGSM
from .mi_fgsm import MIFGSM
from .dsa import DSA


_steps = cfg.attack.steps
_epsilon = cfg.attack.eps
_num_classes = cfg.dataset.num_classes
_budget = cfg.attack.budget


def get_attack(name, *, epsilon=_epsilon, steps=_steps, budget=_budget, local_models=None, num_classes=_num_classes,
               **kwargs):
    attack = eval(
        name + "(local_models=local_models, budget=budget, epsilon=epsilon, num_classes=num_classes, **kwargs)")
    return attack
