"""Create partial metadata.json for katsdptransfer"""

from datetime import datetime
from typing import Sequence, Dict

import numpy as np
import katpoint
import katdal
import astropy.units as u


def target_mask(dataset: katdal.DataSet, target: katpoint.Target) -> np.ndarray:
    """Get boolean mask of dumps where the target is being tracked."""
    # Based on katdal.DataSet.select, but we don't actually use select because
    # we don't want to modify the dataset.
    scan_sensor = dataset.sensor['Observation/scan_state']
    track_mask = (scan_sensor == 'track')
    target_index_sensor = dataset.sensor['Observation/target_index']
    target_index = dataset.catalogue.targets.index(target)
    mask = (target_index_sensor == target_index) & track_mask
    return mask


def time_on_target(dataset: katdal.DataSet, target: katpoint.Target) -> u.Quantity:
    """Seconds spent tracking the target.

    This might be a slight under-report, because dumps that span the
    boundaries are marked as not tracking by katdal, but may have partial
    data that was nevertheless usable.
    """
    mask = target_mask(dataset, target)
    return dataset.dump_period * np.sum(mask) * u.s


def make_metadata(dataset: katdal.DataSet, targets: Sequence[katpoint.Target],
                  channels: int,
                  description: str) -> Dict[str, object]:
    obs_params = dataset.obs_params
    return {
        'CaptureBlockId': dataset.source.capture_block_id,
        'ScheduleBlockIdCode': obs_params.get('sb_id_code', 'UNKNOWN'),
        'Description': obs_params.get('description', 'UNKNOWN') + ': ' + description,
        'ProposalId': obs_params.get('proposal_id', 'UNKNOWN'),
        'Observer': obs_params.get('observer', 'UNKNOWN'),
        # Solr doesn't accept +00:00, only Z, so we can't just format a timezone-aware value
        'StartTime': datetime.utcnow().isoformat() + 'Z',
        'Bandwidth': dataset.channel_width * channels,
        'ChannelWidth': dataset.channel_width,
        'NumFreqChannels': channels,
        'RightAscension': [str(target.radec()[0]) for target in targets],
        'Declination': [str(target.radec()[1]) for target in targets],
        # JSON schema limits format to fixed-point with at most 10 decimal places
        'DecRa': [
            ','.join('{:.10f}'.format(a) for a in np.rad2deg(target.radec())[::-1])
            for target in targets
        ],
        'Targets': [target.name for target in targets],
        'KatpointTargets': [target.description for target in targets],
        # metadata value needs to be in hours
        'IntegrationTime': [time_on_target(dataset, target).to_value(u.h)
                            for target in targets]
    }
