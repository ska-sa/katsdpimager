<%include file="/port.mako"/>
<%include file="atomic.mako"/>
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

<%wg_reduce:define_scratch type="float" size="${wgs_x * wgs_y}" scratch_type="scratch_t" allow_shuffle="${True}"/>
<%wg_reduce:define_function type="float" size="${wgs_x * wgs_y}" function="reduce" scratch_type="scratch_t" broadcast="${False}" allow_shuffle="${True}"/>

KERNEL REQD_WORK_GROUP_SIZE(${wgs_x}, ${wgs_y}, 1)
void mean_weight(
    GLOBAL atomic_accum_float * RESTRICT sums,
    const GLOBAL float * RESTRICT grid,
    int grid_row_stride,
    int grid_width,
    int grid_height)
{
    LOCAL_DECL scratch_t scratch;
    int u = get_global_id(0);
    int v = get_global_id(1);
    int lid = get_local_id(1) * ${wgs_x} + get_local_id(0);
    float w;
    if (u < grid_width && v < grid_height)
        w = grid[v * grid_row_stride + u];
    else
        w = 0;
    float wg_sum = reduce(w, lid, &scratch);
    float wg_sum2 = reduce(w * w, lid, &scratch);
    if (lid == 0)
    {
        atomic_accum_float_add(&sums[0], wg_sum);
        atomic_accum_float_add(&sums[1], wg_sum2);
    }
}
