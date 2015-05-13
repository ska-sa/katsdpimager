<%include file="/port.mako"/>
<%include file="atomic.mako"/>

typedef ${real_type} Real;
typedef ${real_type}2 Complex;
typedef atomic_accum_${real_type}2 atomic_accum_Complex;
#define BATCH_SIZE ${wgs_x * wgs_y}
#define NPOLS ${num_polarizations}
#define TILE_X ${multi_x * wgs_x}
#define TILE_Y ${multi_y * wgs_y}
#define MULTI_X ${multi_x}
#define MULTI_Y ${multi_y}
#define CONVOLVE_KERNEL_OVERSAMPLE ${convolve_kernel_oversample}
#define CONVOLVE_KERNEL_OVERSAMPLE_MASK ${convolve_kernel_oversample - 1}
#define CONVOLVE_KERNEL_OVERSAMPLE_SHIFT ${int.bit_length(convolve_kernel_oversample) - 1}
#define CONVOLVE_KERNEL_SLICE_STRIDE ${convolve_kernel_slice_stride}
#define CONVOLVE_KERNEL_ROW_STRIDE ${convolve_kernel_row_stride}
#define make_Complex make_${real_type}2
#define atomic_accum_Complex_add atomic_accum_${real_type}2_add

DEVICE_FN Complex Complex_mad(Complex a, Complex b, Complex c)
{
    Complex out;
    out.x = fma(a.x, b.x, fma(-a.y, b.y, c.x));
    out.y = fma(a.x, b.y, fma(a.y, b.x, c.y));
    return out;
}

/**
 * Finds smallest value such that ans % tile_size == pos and
 * ans >= low. It requires that tile_size be a power of 2.
 */
DEVICE_FN int wrap(int low, int tile_size, int pos)
{
    return ((low - pos + tile_size - 1) & ~(tile_size - 1)) + pos;
}

DEVICE_FN void writeback(
    GLOBAL atomic_accum_Complex * RESTRICT out, int out_stride,
    int u, int v, const Complex values[NPOLS])
{
    int offset = (v * out_stride + u) * NPOLS;
    for (int p = 0; p < NPOLS; p++)
        atomic_accum_Complex_add(&out[offset + p], values[p]);
}

KERNEL REQD_WORK_GROUP_SIZE(${wgs_x}, ${wgs_y}, 1)
void grid(
    GLOBAL atomic_accum_Complex * RESTRICT out,
    int out_stride,
    const GLOBAL float * RESTRICT uvw,
    const GLOBAL Complex * RESTRICT vis,
    const GLOBAL Complex * RESTRICT convolve_kernel,
    float uv_scale,
    float uv_bias,
    int vis_per_workgroup,
    int nvis)
{
    /* Visibilities are loaded in batches, using (ideally) all workitems in
     * the workgroup to do collective loading, and preprocessed into these
     * arrays.
     */

    LOCAL_DECL int batch_offset[BATCH_SIZE];
    // Index for first grid point to update
    LOCAL_DECL int2 batch_min_uv[BATCH_SIZE];
    // Visibilities multiplied by weights
    LOCAL_DECL Complex batch_vis[NPOLS][BATCH_SIZE];

    // In-register sums
    Complex sums[NPOLS][MULTI_Y][MULTI_X];
    // Last-known UV coordinates
    int2 cur_uv[MULTI_Y][MULTI_X];

    // Zero-initialize things
    for (int y = 0; y < MULTI_Y; y++)
        for (int x = 0; x < MULTI_X; x++)
        {
            for (int p = 0; p < NPOLS; p++)
                sums[y][x][p] = make_Complex(0.0f, 0.0f);
            cur_uv[y][x] = make_int2(0, 0);
        }

    const int lid = get_local_id(1) * ${wgs_x} + get_local_id(0);
    int batch_start = get_group_id(0) * vis_per_workgroup;
    int batch_end = min(nvis, batch_start + vis_per_workgroup);
    for (; batch_start < batch_end; batch_start += BATCH_SIZE)
    {
        // Load batch
        int batch_size = min(nvis - batch_end, BATCH_SIZE);
        if (lid < batch_size)
        {
            int vis_id = batch_start + lid;
            float sample_u = uvw[vis_id * 3];
            float sample_v = uvw[vis_id * 3 + 1];
            // TODO: make portable __float2int_rd
            int fine_u = __float2int_rd(sample_u * uv_scale + uv_bias);
            int fine_v = __float2int_rd(sample_v * uv_scale + uv_bias);
            // TODO: consider scaling and masking v differently to eliminate
            // multiplication by CONVOLVE_KERNEL_OVERSAMPLE later on.
            int sub_u = fine_u & CONVOLVE_KERNEL_OVERSAMPLE_MASK;
            int sub_v = fine_v & CONVOLVE_KERNEL_OVERSAMPLE_MASK;
            int min_u = fine_u >> CONVOLVE_KERNEL_OVERSAMPLE_SHIFT;
            int min_v = fine_v >> CONVOLVE_KERNEL_OVERSAMPLE_SHIFT;

            for (int p = 0; p < NPOLS; p++)
            {
                // TODO: could improve this using float4s where appropriate
                int idx = vis_id * NPOLS + p;
                batch_vis[p][lid] = vis[idx];
            }
            batch_min_uv[lid] = make_int2(min_u, min_v);
            int slice = sub_v * CONVOLVE_KERNEL_OVERSAMPLE + sub_u;
            batch_offset[lid] = slice * CONVOLVE_KERNEL_SLICE_STRIDE
                - (min_v * CONVOLVE_KERNEL_ROW_STRIDE + min_u);
        }

        BARRIER();

        // Process batch
        for (int vis_id = 0; vis_id < batch_size; vis_id++)
        {
            Complex sample_vis[NPOLS];
            for (int p = 0; p < NPOLS; p++)
                sample_vis[p] = batch_vis[p][vis_id];
            int2 min_uv = batch_min_uv[vis_id];
            int base_offset = batch_offset[vis_id];
            for (int y = 0; y < MULTI_Y; y++)
                for (int x = 0; x < MULTI_X; x++)
                {
                    // TODO: expand convolution kernel footprint such that
                    // multi-xy block is emitted as a unit, instead of separate
                    // cur_uv for each element.
                    int u = wrap(min_uv.x, TILE_X, MULTI_X * get_local_id(0) + x);
                    int v = wrap(min_uv.y, TILE_Y, MULTI_Y * get_local_id(1) + y);
                    if (u != cur_uv[y][x].x || v != cur_uv[y][x].y)
                    {
                        writeback(out, out_stride, cur_uv[y][x].x, cur_uv[y][x].y, sums[y][x]);
                        cur_uv[y][x] = make_int2(u, v);
                        for (int p = 0; p < NPOLS; p++)
                            sums[y][x][p] = make_Complex(0.0f, 0.0f);
                    }
                    int offset = v * CONVOLVE_KERNEL_ROW_STRIDE + u + base_offset;
                    Complex weight = convolve_kernel[offset];
                    for (int p = 0; p < NPOLS; p++)
                        sums[y][x][p] = Complex_mad(weight, sample_vis[p], sums[y][x][p]);
                }
        }

        // Necessary to prevent the next batch overwriting data for the current
        // one while still in progress
        BARRIER();
    }
}
