"""
Dataframe-like class to hold lichtblick pf structure.
"""

from __future__ import annotations
from .singlepf_multipf import SinglePf, MultiPf
from typing import Union, Iterable


class LbPf(MultiPf):
    """Lichtblick Portfolio.

    Parameters
    ----------
    data : Iterable[Union[LbPf, MultiPf]], optional
        Iterable of children of this portfolio.
    offtake : Union[SinglePf, MultiPf], optional
        The offtake for this portfolio - at this pf level, so excluding its children.
    sourced : Union[SinglePf, MultiPf], optional
        The procurement for this portfolio - at this pf level, so excluding its children.
    name : str
        Name of this portfolio.

    Attributes
    ----------
    Offtake : SinglePf
        Sum of all offtake at this pf level and at all its children.
    Sourced: SinglePf
        Sum of all procurement at this pf level and at all its children.
    Unhedged: SinglePf
        Sum of offtake and sourced.

    Notes
    -----
    A lichtblick portfolio is a multipf with a specific structure:
    . Its children are other LichtblickPf objects.
    . One its children may be a MultiPf called 'Own', where offtake and procurement 
      on this pf level are specified.
    . The 'Own' Portfolio has at most 2 children: Offtake and Sourced, each of which is
      a SinglePf or MultiPf. 
    Aggregated offtake and procurement (for the portfolio and its children) can be 
    accessed through .Offtake and .Sourced.
    """

    def __init__(
        self,
        data: Iterable[Union[LbPf, MultiPf]] = None,
        offtake: Union[SinglePf, MultiPf] = None,
        sourced: Union[SinglePf, MultiPf] = None, *,
        name: str = None,
    ):
        own = MultiPf(name="Own")
        if offtake:
            offtake.name = "Offtake"
            own.add_child(offtake)
        if sourced:
            sourced.name = "Sourced"
            own.add_child(sourced)
        if offtake or sourced:
            super().__init__([own], name=name)
        else:
            super().__init__(name=name)

    @property
    def Offtake(self) -> SinglePf:
        return self._sumifexists('Offtake')

    @property
    def Sourced(self) -> SinglePf:
        return self._sumifexists('Sourced')

    @property
    def Unhedged(self) -> SinglePf:
        return SinglePf({'q': self.Offtake.q + self.Sourced.q}, name='Unhedged')

    def _sumifexists(self, attr):
        tosum = []
        for child in self:
            try:
                tosum.append(getattr(child, attr))
            except AttributeError:
                pass
        return sum(tosum)
        


