#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import casacore.tables
import pyfits
import katsdpsigproc
import katsdpimager.loader
from contextlib import closing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, metavar='INPUT', help='Input measurement set')
    parser.add_argument('output_file', type=str, metavar='OUTPUT', help='Output FITS file')
    parser.add_argument('--channel', '-c', type=int, default=0, help='Channel number')

    args = parser.parse_args()
    print("Converting {} to {}".format(args.input_file, args.output_file))
    with closing(katsdpimager.loader.load(args.input_file)) as dataset:
        print(dataset.antenna_diameter())
        print(dataset.longest_baseline())
        for chunk in dataset.data_iter(args.channel, 65536):
            print(chunk['vis'].shape, chunk['weights'].shape, chunk['uvw'].shape)

if __name__ == '__main__':
    main()
