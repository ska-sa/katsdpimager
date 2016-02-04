<%include file="/port.mako"/>

#define NPOLS ${num_polarizations}

KERNEL REQD_WORK_GROUP_SIZE(${wgs_x}, ${wgs_y}, 1)
void density_weights(
    GLOBAL float * RESTRICT grid,
    int grid_row_stride,
    int grid_pol_stride,
    float a, float b)
{
    int u = get_global_id(0);
    int v = get_global_id(1);
    int address = v * grid_row_stride + u;
    for (int pol = 0; pol < NPOLS; pol++)
    {
        float w = grid[address];
        if (w != 0.0)
        {
            w = 1.0 / fma(a, w, b);
            grid[address] = w;
        }
        address += grid_pol_stride;
    }
}
