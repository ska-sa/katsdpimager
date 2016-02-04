<%include file="/port.mako"/>
<%include file="atomic.mako"/>

#define NPOLS ${num_polarizations}

KERNEL REQD_WORK_GROUP_SIZE(${wgs}, 1, 1)
void grid_weights(
    GLOBAL atomic_accum_float * RESTRICT grid,
    int grid_row_stride,
    int grid_pol_stride,
    const GLOBAL short2 * RESTRICT uv,
    const GLOBAL float * RESTRICT weights,
    int address_bias,
    int num_vis)
{
    int gid = get_global_id(0);
    if (gid >= num_vis)
        return;
    float sample_weights[NPOLS];
    short2 sample_uv = uv[2 * gid];   // 2 * because we're really taking first two elements from short4
    for (int pol = 0; pol < NPOLS; pol++)
        sample_weights[pol] = weights[gid * NPOLS + pol];
    int address = sample_uv.y * grid_row_stride + sample_uv.x + address_bias;
    for (int pol = 0; pol < NPOLS; pol++)
    {
        atomic_accum_float_add(&grid[address], sample_weights[pol]);
        address += grid_pol_stride;
    }
}
