<%include file="/port.mako"/>
<%include file="atomic.mako"/>
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

<%wg_reduce:define_scratch type="float" size="${wgs_x * wgs_y}" scratch_type="scratch_t" allow_shuffle="${True}"/>
<%wg_reduce:define_function type="float" size="${wgs_x * wgs_y}" function="reduce" scratch_type="scratch_t" broadcast="${False}" allow_shuffle="${True}"/>

#define NPOLS ${num_polarizations}

KERNEL REQD_WORK_GROUP_SIZE(${wgs_x}, ${wgs_y}, 1)
void density_weights(
    GLOBAL atomic_accum_float * RESTRICT sums,
    GLOBAL float * RESTRICT grid,
    int grid_row_stride,
    int grid_pol_stride,
    int size_x, int size_y,
    float a, float b)
{
    LOCAL_DECL scratch_t scratch;
    int u = get_global_id(0);
    int v = get_global_id(1);
    int lid = get_local_id(1) * ${wgs_x} + get_local_id(0);
    int address = v * grid_row_stride + u;
    bool inside = (u < size_x && v < size_y);
    for (int pol = 0; pol < NPOLS; pol++)
    {
        float w = inside ? grid[address] : 0.0f;
        float d = (w != 0.0f) ? 1.0f / fma(a, w, b) : 0.0f;
        if (pol == 0)
        {
            float dw = d * w;
            float sum_w = reduce(w, lid, &scratch);
            float sum_dw = reduce(dw, lid, &scratch);
            float sum_d2w = reduce(d * dw, lid, &scratch);
            if (lid == 0)
            {
                atomic_accum_float_add(&sums[0], sum_w);
                atomic_accum_float_add(&sums[1], sum_dw);
                atomic_accum_float_add(&sums[2], sum_d2w);
            }
        }
        grid[address] = d;
        address += grid_pol_stride;
    }
}
