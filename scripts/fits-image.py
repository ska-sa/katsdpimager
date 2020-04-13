#!/usr/bin/env python3

import argparse

import katsdpimager.render as render


def main():
    parser = argparse.ArgumentParser(
        description='Create image from FITS file written by katsdpimager.'
    )
    parser.add_argument('--width', type=int, default=1024,
                        help='Image width [%(default)s]')
    parser.add_argument('--height', type=int, default=768,
                        help='Image height [%(default)s]')
    parser.add_argument('--dpi', type=float, default=render.DEFAULT_DPI,
                        help='Dots per inch [%(default)s]')
    parser.add_argument('--caption', type=str,
                        help='Caption to add to the image')
    parser.add_argument('input', help='Input FITS file')
    parser.add_argument('output', help='Output image file')
    args = parser.parse_args()

    render.write_image(args.input, args.output, width=args.width, height=args.height)


if __name__ == '__main__':
    main()
