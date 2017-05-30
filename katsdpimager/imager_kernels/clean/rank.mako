<%include file="/port.mako"/>
<%include file="metric.mako"/>
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

${wg_reduce.define_scratch('uint', wgsx * wgsy, 'reduce_scratch', allow_shuffle=True)}
${wg_reduce.define_function('uint', wgsx * wgsy, 'reduce', 'reduce_scratch',
                            allow_shuffle=True, broadcast=False)}

#define WGSX ${wgsx}
#define WGSY ${wgsy}
#define TILEX ${tilex}
#define TILEY ${tiley}
#define NPOLS ${num_polarizations}
typedef ${real_type} T;

KERNEL REQD_WORK_GROUP_SIZE(WGSX, WGSY, 1)
void compute_rank(
    const GLOBAL T * RESTRICT image,
    int image_row_stride,
    int image_pol_stride,
    int image_width,
    int image_height,
    int image_border,
    GLOBAL uint * RESTRICT rank,
    T value)
{
    LOCAL_DECL reduce_scratch scratch;

    int x0 = get_global_id(0) * (TILEX / WGSX) + image_border;
    int y0 = get_global_id(1) * (TILEY / WGSY) + image_border;
    uint local_rank = 0;
    for (int i = 0; i < TILEY / WGSY; i++)
        for (int j = 0; j < TILEX / WGSX; j++)
        {
            int y = y0 + i;
            int x = x0 + j;
            if (x < image_width && y < image_height)
            {
                int pixel_offset = y * image_row_stride + x;
                T metric = clean_metric(image, pixel_offset, image_pol_stride);
                local_rank += (metric < value);
            }
        }

    int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    uint wg_rank = reduce(local_rank, lid, &scratch);
    if (lid == 0)
    {
        int idx = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        rank[idx] = wg_rank;
    }
}
