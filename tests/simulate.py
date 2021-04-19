#!/usr/bin/env python3

"""
Create a simulated Measurement Set for use in comparing a number of imagers
against each other.

The time range, frequencies and phase centre are hard-coded in the script,
while the local sky model (consisting of point sources with polarisation but
no spectral shape) is read from lsm.txt, and the antennas from
meerkat_antennas.txt.
"""

import os

import numpy as np
import pandas
import katpoint
import casacore.tables

import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
import astropy.io.ascii

# RASCIL gets upset if it can't find its data directory, but doesn't actually
# need it for this simulation. Just give it a fake one.
if 'RASCIL_DATA' not in os.environ:
    os.environ['RASCIL_DATA'] = '/'

import rascil.data_models                   # noqa: E402
import rascil.processing_components         # noqa: E402


freqs = 856e6 + 214e6 * (np.arange(4) + 0.5)
start_time = Time('2014-01-01T15:00:00.0', format='isot', scale='utc')
time_step = 4.0 * u.s
times = start_time + time_step * (np.arange(3600) + 0.5)
phase_centre = SkyCoord(ra='3h30m00s', dec='-35d00m00s', frame='icrs')

lsm = astropy.io.ascii.read('lsm.txt')
comps = [
    rascil.data_models.Skycomponent(
        SkyCoord(
            ra=Angle(row['ra'], unit='hour'),
            dec=Angle(row['dec'], unit='deg'),
            frame='icrs'
        ),
        [1e9],       # Arbitrary frequency
        flux=[[row['I'], row['Q'], row['U'], row['V']]]
    )
    for row in lsm
]

antennas = []
with open('meerkat_antennas.txt') as f:
    for line in f:
        if line:
            antennas.append(katpoint.Antenna(line))
location = antennas[0].ref_location
config = rascil.data_models.Configuration(
    location=location,
    names=[ant.name for ant in antennas],
    mount=np.repeat('azel', len(antennas)),
    xyz=[list(ant.position_enu) for ant in antennas],
    vp_type=np.repeat('unknown', len(antennas)),
    diameter=[ant.diameter.to_value(u.m) for ant in antennas],
    name='MeerKAT'
)

# RASCIL doesn't have very accurate UVW calculations
# (see https://gitlab.com/ska-telescope/rascil/-/issues/43), so we use
# katpoint instead.
baselines = pandas.MultiIndex.from_arrays(
    np.triu_indices(len(antennas)),
    names=('antenna1', 'antenna2')
)
ref_ant = antennas[0].array_reference_antenna()
target = katpoint.Target(phase_centre)
uvw_ant = target.uvw(antennas, times, ref_ant)
uvw = (uvw_ant[..., baselines.get_level_values('antenna1')]
       - uvw_ant[..., baselines.get_level_values('antenna2')])

vis_shape = (len(times), len(baselines), len(freqs), 4)
times_mjds = times.to_value('mjd') * 86400.0
bvis = rascil.data_models.BlockVisibility(
    frequency=freqs,
    channel_bandwidth=np.repeat(freqs[1] - freqs[0], len(freqs)),
    phasecentre=phase_centre,
    configuration=config,
    uvw=np.moveaxis(uvw, 0, -1),     # Move u/v/w axis to the end
    time=times_mjds,
    vis=np.zeros(vis_shape, np.complex64),
    integration_time=np.repeat(time_step.to_value(u.s), len(times)),
    flags=np.zeros(vis_shape, np.int_),
    baselines=baselines,
    polarisation_frame=rascil.data_models.PolarisationFrame('linear')
)
# Run the simulation
bvis = rascil.processing_components.dft_skycomponent_visibility(bvis, comps)
# Write the output
rascil.processing_components.export_blockvisibility_to_ms('simple.ms', [bvis])

# Make some fixes to the tables. See
# - https://gitlab.com/ska-telescope/rascil/-/issues/42
# - https://gitlab.com/ska-telescope/rascil/-/issues/44
with casacore.tables.table('simple.ms', readonly=False, ack=False) as t:
    timestamps = np.repeat(times_mjds, len(baselines))
    t.putcol('TIME', timestamps)
    t.putcol('TIME_CENTROID', timestamps)
with casacore.tables.table('simple.ms::ANTENNA', readonly=False, ack=False) as t:
    xyz = [u.Quantity(ant.location.geocentric).to_value(u.m) for ant in antennas]
    t.putcol('POSITION', np.array(xyz))
    t.putcol('MOUNT', ['ALT-AZ'] * len(antennas))
