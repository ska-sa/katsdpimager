<%include file="/port.mako"/>
<%namespace name="sample" file="sample.mako"/>

#define WGSX ${wgsx}
#define WGSY ${wgsy}
#define NPOLS ${num_polarizations}
typedef ${real_type} T;

${sample.define_sample(real_type, wgsx * wgsy)}

/**
 * Find the global peak from per-tile peaks. Only a single workgroup is used.
 */
KERNEL REQD_WORK_GROUP_SIZE(WGSX, WGSY, 1)
void find_peak(
    const GLOBAL T * RESTRICT dirty,
    int image_row_stride,
    int image_pol_stride,
    const GLOBAL T * RESTRICT tile_max,
    const GLOBAL int2 * RESTRICT tile_pos,
    int tile_stride,
    int tile_width,
    int tile_height,
    GLOBAL T * RESTRICT peak_value,
    GLOBAL int2 * RESTRICT peak_pos,
    GLOBAL T * RESTRICT peak_pixel)
{
    LOCAL_DECL reduce_sample_max_scratch scratch;

    sample priv;
    priv.value = -1;
    for (int y = get_local_id(1); y < tile_height; y += WGSY)
        for (int x = get_local_id(0); x < tile_width; x += WGSX)
        {
            int idx = y * tile_stride + x;
            T value = tile_max[idx];
            if (value > priv.value)
            {
                priv.value = value;
                priv.pos = tile_pos[idx];
            }
        }

    int linear_id = get_local_id(1) * WGSX + get_local_id(0);
    sample best = reduce_sample_max(priv, linear_id, &scratch);
    if (linear_id == 0)
    {
        *peak_value = best.value;
        *peak_pos = best.pos;
        // Note: .x here is actually y, since pos is stored as y, x
        for (int pol = 0; pol < NPOLS; pol++)
            peak_pixel[pol] = dirty[pol * image_pol_stride + best.pos.x * image_row_stride + best.pos.y];
    }
}
