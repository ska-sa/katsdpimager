#!/usr/bin/env python

"""Generates a C module for sorting visibilities in the preprocessor. It is
slightly complicated by the need to support 1-4 polarizations while still
presenting a C-compatible interface. The templating is thus all done with
Python generating the code."""

from __future__ import print_function, division
import cffi

ffi = cffi.FFI()
source = """
#include <algorithm>
#include <stdint.h>

namespace
{
"""

for i in range(1, 5):
    struct = """
        typedef struct
        {{
            int16_t uv[2];
            int16_t sub_uv[2];
            float weights[{N}];
            float vis[{N}][2];
            int16_t w_plane;
            int16_t w_slice;
            int32_t channel;
            int32_t baseline;
        }} vis_{N}_t;
        """.format(N=i)
    ffi.cdef(struct)
    ffi.cdef("void sort_vis_{N}(vis_{N}_t *vis, size_t N);".format(N=i))
    source += struct

source += """
    template<typename T>
    struct compare
    {
        bool operator()(const T &a, const T &b) const
        {
            if (a.channel != b.channel)
                return a.channel < b.channel;
            else if (a.w_slice != b.w_slice)
                return a.w_slice < b.w_slice;
            else
                return a.baseline < b.baseline;
        }
    };
} // anonymous namespace

extern "C"
{
"""
for i in range(1, 5):
    source += """
        static void sort_vis_{N}(vis_{N}_t *vis, size_t N)
        {{
            std::stable_sort(vis, vis + N, compare<vis_{N}_t>());
        }}""".format(N=i)
source += """
} // extern "C"
"""
ffi.set_source("_sort_vis", source, source_extension='.cpp')
if __name__ == '__main__':
    ffi.compile()
