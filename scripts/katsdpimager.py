#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import casacore.tables
import pyfits
import katsdpsigproc
import katsdpimager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, metavar='INPUT', help='Input measurement set')
    parser.add_argument('output_file', type=str, metavar='OUTPUT', help='Output FITS file')

    args = parser.parse_args()
    print("Converting {} to {}".format(args.input_file, args.output_file))

if __name__ == '__main__':
    main()
