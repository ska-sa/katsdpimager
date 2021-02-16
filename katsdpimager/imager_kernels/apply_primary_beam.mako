## Divide an image by the primary beam. Where the beam power is less
## than a threshold, set a specific value instead.
##
## The primary beam is currently polarization-independent.

<%include file="/port.mako"/>

typedef ${real_type} T;
#define NPOLS ${num_polarizations}

KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1)
void apply_primary_beam(
    GLOBAL T * RESTRICT image,
    const GLOBAL T * RESTRICT beam_power,
    int row_stride,
    int pol_stride,
    int width, int height,
    T threshold,
    T replacement)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height)
        return;
    int addr = y * row_stride + x;
    T beam = beam_power[addr];
    T values[NPOLS];
    if (beam < threshold)
    {
        for (int i = 0; i < NPOLS; i++)
            values[i] = replacement;
    }
    else
    {
        for (int i = 0; i < NPOLS; i++)
            values[i] = image[addr + i * pol_stride];
        // Split into two loops to better hide latency (untested)
        for (int i = 0; i < NPOLS; i++)
            values[i] /= beam;
    }
    for (int i = 0; i < NPOLS; i++)
        image[addr + i * pol_stride] = values[i];
}
