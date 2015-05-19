<%include file="/port.mako"/>
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

#define WGSX ${wgsx}
#define WGSY ${wgsy}
#define TILEX ${tilex}
#define TILEY ${tiley}
#define NPOLS ${num_polarizations}
typedef ${real_type} T;

typedef struct
{
    T v[NPOLS];
} pixel;

typedef struct
{
    T value;
    int2 pos;
} sample;

DEVICE_FN sample sample_max(sample a, sample b)
{
    return a.value > b.value ? a : b;
}

<%def name="sample_max(a, b, type)">sample_max((${a}), (${b}))</%def>
${wg_reduce.define_scratch('sample', wgsx * wgsy, 'reduce_sample_max_scratch')}
${wg_reduce.define_function('sample', wgsx * wgsy, 'reduce_sample_max', 'reduce_sample_max_scratch', sample_max)}

KERNEL REQD_WORK_GROUP_SIZE(WGSX, WGSY, 1)
void update_tiles(
    const GLOBAL pixel * RESTRICT image,
    int image_stride,
    int image_width,
    int image_height,
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
    int x0 = tile_x * TILEX + get_local_id(0);
    int y0 = tile_y * TILEY + get_local_id(1);
    for (int y = 0; y < TILEY; y += WGSY)
        for (int x = 0; x < TILEX; x += WGSX)
        {
            T value;
            int image_x = x0 + x;
            int image_y = y0 + y;
            if (image_x < image_width && image_y < image_height)
            {
                int pixel_offset = image_y * image_stride + image_x;
                pixel pix = image[pixel_offset];
% if clean_sumsq:
                value = pix.v[0] * pix.v[0];
                for (int i = 1; i < NPOLS; i++)
                    value = fma(pix.v[i], pix.v[i], value);
% else:
                value = fabs(pix.v[0]);
% endif
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
