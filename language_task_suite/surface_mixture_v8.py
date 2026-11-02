"""Clean-line section headers for the v7 scoped-fact experiment.

V7 inherits generic numbered narrative lines, so some section headings were
rendered with two different numbers. V8 starts a Section only after a newline
and rules it out after a generic numbered-line prefix. Other numbered lines
remain in the mixture.
"""

from surface_mixture_v2 import _split
from surface_mixture_v7 import SurfaceMixtureV7


class SurfaceMixtureV8(SurfaceMixtureV7):
    @staticmethod
    def _without_section(out):
        section_mass = out.pop('Section', 0.)
        if section_mass:
            residual = sum(mass for token, mass in out.items() if token != '\n')
            if residual:
                factor = (residual+section_mass)/residual
                out = {token: (mass if token == '\n' else mass*factor)
                       for token, mass in out.items()}
        return out

    def _start_distribution(self, history, after_newline):
        out = super()._start_distribution(history, after_newline)
        if after_newline:
            return out
        return self._without_section(out)

    def next_distribution(self, prefix):
        _, raw = _split(self._tokens(prefix))
        if (len(raw) == 3 and raw[0] == '\n' and raw[1].isdigit()
                and raw[2] == ':'):
            return self._without_section(super().next_distribution(prefix))
        return super().next_distribution(prefix)
