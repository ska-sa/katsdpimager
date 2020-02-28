#!/usr/bin/env python3

import argparse

import katsdpimager.render as render


def main():
    parser = argparse.ArgumentParser(
        description='Create video from FITS files written by katsdpimager.'
    )
    parser.add_argument('--width', type=int, default=1024,
                        help='Video width [%(default)s]')
    parser.add_argument('--height', type=int, default=768,
                        help='Video height [%(default)s]')
    parser.add_argument('--fps', type=float, default=5.0,
                        help='Frames per second [%(default)s]')
    parser.add_argument('output', help='Output video file')
    parser.add_argument('input', nargs='+', help='Input FITS files')
    args = parser.parse_args()

    files = [(None, arg) for arg in args.input]
    render.write_movie(files, args.output, width=args.width, height=args.height, fps=args.fps)


if __name__ == '__main__':
    main()
