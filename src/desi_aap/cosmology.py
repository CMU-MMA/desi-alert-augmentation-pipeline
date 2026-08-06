"""Cosmology models shared by the TNS catalog and GraceDB crossmatch stages."""

from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, Planck18

SHOES_H0 = 73.04
SHOES_OM0 = 0.3
CMB_TEMPERATURE_K = 2.725
SHOES = FlatLambdaCDM(H0=SHOES_H0, Om0=SHOES_OM0, Tcmb0=CMB_TEMPERATURE_K * u.K)
COSMOLOGIES = {
    "SHOES": SHOES,
    "Planck18": Planck18,
}
