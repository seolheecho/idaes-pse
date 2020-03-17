##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Methods for ideal equations of state.

Currently only supports liquid and vapor phases
"""
from idaes.core.util.exceptions import PropertyNotSupportedError
from idaes.generic_models.properties.core.generic.generic_property import (
    get_method, get_component_object as cobj)
from idaes.core.util.constants import Constants as const


# TODO: Add support for ideal solids
def common(b):
    # No common components required for ideal property calculations
    pass


def dens_mass_phase(b, p):
    return b.dens_mol_phase[p]*b.mw_phase[p]


def dens_mol_phase(b, p):
    pobj = b.params.get_phase(p)
    if pobj.is_vapor_phase():
        return b.pressure/(const.gas_constant*b.temperature)
    elif pobj.is_liquid_phase():
        return sum(b.mole_frac_phase_comp[p, j] *
                   get_method(b, "dens_mol_liq_comp", j)(
                       b, cobj(b, j), b.temperature)
                   for j in b.components_in_phase(p))
    else:
        raise PropertyNotSupportedError(_invalid_phase_msg(b.name, p))


def enth_mol_phase(b, p):
    return sum(b.mole_frac_phase_comp[p, j]*b.enth_mol_phase_comp[p, j]
               for j in b.components_in_phase(p))


def enth_mol_phase_comp(b, p, j):
    pobj = b.params.get_phase(p)
    if pobj.is_vapor_phase():
        return get_method(b, "enth_mol_ig_comp", j)(
            b, cobj(b, j), b.temperature)
    elif pobj.is_liquid_phase():
        return get_method(b, "enth_mol_liq_comp", j)(
            b, cobj(b, j), b.temperature)
    else:
        raise PropertyNotSupportedError(_invalid_phase_msg(b.name, p))


def entr_mol_phase(b, p):
    return sum(b.mole_frac_phase_comp[p, j]*b.entr_mol_phase_comp[p, j]
               for j in b.components_in_phase(p))


def entr_mol_phase_comp(b, p, j):
    pobj = b.params.get_phase(p)
    if pobj.is_vapor_phase():
        return get_method(b, "entr_mol_ig_comp", j)(
            b, cobj(b, j), b.temperature)
    elif pobj.is_liquid_phase():
        return get_method(b, "entr_mol_liq_comp", j)(
            b, cobj(b, j), b.temperature)
    else:
        raise PropertyNotSupportedError(_invalid_phase_msg(b.name, p))


def fug_phase_comp(b, p, j):
    pobj = b.params.get_phase(p)
    if pobj.is_vapor_phase():
        return b.mole_frac_phase_comp[p, j]*b.pressure
    elif pobj.is_liquid_phase():
        return b.mole_frac_phase_comp[p, j] * \
               get_method(b, "pressure_sat_comp", j)(b, cobj(b, j), b._teq)
    else:
        raise PropertyNotSupportedError(_invalid_phase_msg(b.name, p))


def fug_coeff_phase_comp(b, p, j):
    pobj = b.params.get_phase(p)
    if not (pobj.is_vapor_phase() or pobj.is_liquid_phase()):
        raise PropertyNotSupportedError(_invalid_phase_msg(b.name, p))
    return 1


def gibbs_mol_phase(b, p):
    return sum(b.mole_frac_phase_comp[p, j]*b.gibbs_mol_phase_comp[p, j]
               for j in b.components_in_phase(p))


def gibbs_mol_phase_comp(b, p, j):
    return (b.enth_mol_phase_comp[p, j] -
            b.entr_mol_phase_comp[p, j] *
            b.temperature)


def _invalid_phase_msg(name, phase):
    return ("{} received unrecognised phase name {}. Ideal property "
            "libray only supports Vap and Liq phases."
            .format(name, phase))
