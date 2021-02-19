## Add one image to another

<%include file="/port.mako"/>

typedef ${real_type} T;
#define NPOLS ${num_polarizations}

KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1)
void add_image(
    GLOBAL T * RESTRICT dest,
    GLOBAL const T * RESTRICT src,
    int dest_row_stride,
    int dest_pol_stride,
    int src_row_stride,
    int src_pol_stride,
    int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height)
        return;
    int dest_addr = y * dest_row_stride + x;
    int src_addr = y * src_row_stride + x;
    for (int pol = 0; pol < NPOLS; pol++)
    {
        dest[dest_addr] += src[src_addr];
        dest_addr += dest_pol_stride;
        src_addr += src_pol_stride;
    }
}
