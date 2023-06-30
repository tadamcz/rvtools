# Subpackage named 'gen' in keeping with SciPy calling their classes ``norm_gen``, ``beta_gen``, etc
from rvtools.dists.gen.tp_uniform import TwoPieceUniform
from rvtools.dists.gen.halves_uniform import HalvesUniform
from rvtools.dists.gen.certainty import Certainty
from betapert import pert, mpert  # noqa

# These being instances, not classes, is not IMO idiomatic Python, but it's core to the way SciPy's
# ``rv_continuous`` class works. See examples of how SciPy defines their distributions in
# ``scipy/stats/_continuous_distns.py``.
tp_uniform = TwoPieceUniform()
halves_uniform = HalvesUniform()
certainty = Certainty()
