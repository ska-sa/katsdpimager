## Scale the image by a constant amount per polarization

<%include file="/port.mako"/>

typedef ${real_type} T;
#define NPOLS ${num_polarizations}

typedef struct
{
    T value[NPOLS];
} scale_t;

KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1)
void scale(
    GLOBAL T * RESTRICT image,
    int row_stride,
    int pol_stride,
    int width, int height,
    scale_t scale_factor)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height)
        return;
    int addr = y * row_stride + x;
    for (int pol = 0; pol < NPOLS; pol++)
    {
        image[addr] *= scale_factor.value[pol];
        addr += pol_stride;
    }
}
