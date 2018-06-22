<%include file="/port.mako"/>
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

#define WGSX ${wgsx}
#define WGSY ${wgsy}
#define NPOLS ${num_polarizations}
typedef ${real_type} T;

DEVICE_FN int2 max2(int2 a, int2 b)
{
    return make_int2(max(a.x, b.x), max(a.y, b.y));
}

<%def name="op_max2(a, b, type)">(max2((${a}), (${b})))</%def>
${wg_reduce.define_scratch('int2', wgsx * wgsy, 'reduce_scratch', allow_shuffle=True)}
${wg_reduce.define_function('int2', wgsx * wgsy, 'reduce_max_int2', 'reduce_scratch',
                            op_max2, allow_shuffle=True, broadcast=False)}

/**
 * Compute workgroup-local maximum x and y distances from the centre at which
 * values greater than @a threshold occur.
 *
 * @param psf              The point spread function with shape (pols, height, width)
 * @param psf_row_stride   Stride between rows of @a psf
 * @param psf_pol_stride   Stride between slices of @a psf
 * @param[out] bound       Maximum x, y distances, shape (height, width), packed
 * @param min_x, min_y     Minimum coordinates to examine in PSF (used as offsets)
 * @param max_x, max_y     Maximum coordinates to examine in PSF
 * @param mid_x, mid_y     Centre of the PSF
 * @param threshold        Cutoff value above which we wish to find the bounding box
 */
KERNEL REQD_WORK_GROUP_SIZE(WGSX, WGSY, 1)
void psf_patch(
    const GLOBAL T * RESTRICT psf,
    int psf_row_stride,
    int psf_pol_stride,
    GLOBAL int2 * RESTRICT bound,
    int min_x,
    int min_y,
    int max_x,
    int max_y,
    int mid_x,
    int mid_y,
    float threshold)
{
    LOCAL_DECL reduce_scratch scratch;

    int x = min(get_global_id(0) + min_x, max_x);
    int y = min(get_global_id(1) + min_y, max_y);
    int wg_x = get_group_id(0);
    int wg_y = get_group_id(1);
    int addr = y * psf_row_stride + x;
    bool over = false;  // whether (x, y) is over the threshold in any polarization
    for (int i = 0; i < NPOLS; i++, addr += psf_pol_stride)
    {
        over |= fabs(psf[addr]) >= threshold;
    }
    int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int dx = over ? abs(x - mid_x) : 0;
    int dy = over ? abs(y - mid_y) : 0;
    int2 result = reduce_max_int2(make_int2(dx, dy), lid, &scratch);
    if (lid == 0)
        bound[wg_y * get_num_groups(0) + wg_x] = result;
}
