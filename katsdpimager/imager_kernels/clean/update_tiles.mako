<%include file="/port.mako"/>
<%include file="metric.mako"/>
<%namespace name="sample" file="sample.mako"/>

#define WGSX ${wgsx}
#define WGSY ${wgsy}
#define TILEX ${tilex}
#define TILEY ${tiley}
#define NPOLS ${num_polarizations}
typedef ${real_type} T;

${sample.define_sample(real_type, wgsx * wgsy)}

/**
 * Find peak within each tile. One workgroup is launched for each tile. The
 * workgroup may be smaller than the tile (but exactly divide into it), in
 * which case each workitem reduces multiple elements before the
 * workgroup-wide reduction.
 *
 * The image_width and image_height are the indices of the last valid
 * (non-border) pixels i.e. the true width and height less the border width.
 */
KERNEL REQD_WORK_GROUP_SIZE(WGSX, WGSY, 1)
void update_tiles(
    const GLOBAL T * RESTRICT image,
    int image_row_stride,
    int image_pol_stride,
    int image_width,
    int image_height,
    int image_border,
    GLOBAL T * RESTRICT tile_max,
    GLOBAL int2 * RESTRICT tile_pos,
    int tile_stride,
    int tile_offset_x,
    int tile_offset_y)
{
    LOCAL_DECL reduce_sample_max_scratch scratch;

    sample priv;
    int tile_x = get_group_id(0) + tile_offset_x;
    int tile_y = get_group_id(1) + tile_offset_y;
    int x0 = tile_x * TILEX + get_local_id(0) + image_border;
    int y0 = tile_y * TILEY + get_local_id(1) + image_border;
    for (int y = 0; y < TILEY; y += WGSY)
        for (int x = 0; x < TILEX; x += WGSX)
        {
            T value;
            int image_x = x0 + x;
            int image_y = y0 + y;
            if (image_x < image_width && image_y < image_height)
            {
                int pixel_offset = image_y * image_row_stride + image_x;
                value = clean_metric(image, pixel_offset, image_pol_stride);
            }
            else
                value = -1;
            if ((x == 0 && y == 0) || value > priv.value)
            {
                priv.value = value;
                priv.pos = make_int2(image_y, image_x);
            }
        }

    int linear_id = get_local_id(1) * WGSX + get_local_id(0);
    sample best = reduce_sample_max(priv, linear_id, &scratch);
    if (linear_id == 0)
    {
        int idx = tile_y * tile_stride + tile_x;
        tile_max[idx] = best.value;
        tile_pos[idx] = best.pos;
    }
}
