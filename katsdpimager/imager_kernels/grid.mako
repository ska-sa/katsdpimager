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
#define CONVOLVE_KERNEL_SIZE_X ${convolve_kernel_size_x}
#define CONVOLVE_KERNEL_SIZE_Y ${convolve_kernel_size_y}
#define CONVOLVE_KERNEL_OVERSAMPLE ${convolve_kernel_oversample}
#define CONVOLVE_KERNEL_OVERSAMPLE_MASK ${convolve_kernel_oversample - 1}
#define CONVOLVE_KERNEL_OVERSAMPLE_SHIFT ${int.bit_length(convolve_kernel_oversample) - 1}
#define CONVOLVE_KERNEL_SLICE_STRIDE ${convolve_kernel_slice_stride}
#define CONVOLVE_KERNEL_W_STRIDE ${convolve_kernel_w_stride}
#define CONVOLVE_KERNEL_W_SCALE ${convolve_kernel_w_scale}f
#define CONVOLVE_KERNEL_MAX_W ${convolve_kernel_max_w}f
#define make_Complex make_${real_type}2
#define atomic_accum_Complex_add atomic_accum_${real_type}2_add

/// Computes a * b
DEVICE_FN float2 complex_mul(float2 a, float2 b)
{
    return make_float2(fma(a.x, b.x, -a.y * b.y),
                       fma(a.x, b.y, a.y * b.x));
}

/// Computes a * conj(b) + c
DEVICE_FN Complex Complex_madc(float2 a, float2 b, Complex c)
{
    Complex out;
    out.x = fma(a.x, b.x, fma(a.y, b.y, c.x));
    out.y = fma(a.x, -b.y, fma(a.y, b.x, c.y));
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
    GLOBAL atomic_accum_Complex * RESTRICT out,
    int out_row_stride,
    int out_pol_stride,
    int u, int v, const Complex values[NPOLS])
{
    int offset = (v * out_row_stride + u);
    for (int p = 0; p < NPOLS; p++)
    {
        atomic_accum_Complex_add(&out[offset], values[p]);
        offset += out_pol_stride;
    }
}

KERNEL REQD_WORK_GROUP_SIZE(${wgs_x}, ${wgs_y}, 1)
void grid(
    GLOBAL atomic_accum_Complex * RESTRICT out,
    int out_row_stride,
    int out_pol_stride,
    const GLOBAL float * RESTRICT uvw,
    const GLOBAL float2 * RESTRICT vis,
    const GLOBAL float2 * RESTRICT convolve_kernel,
    float uv_scale,
    float uv_bias,
    int vis_per_workgroup,
    int nvis)
{
    /* Visibilities are loaded in batches, using (ideally) all workitems in
     * the workgroup to do collective loading, and preprocessed into these
     * arrays.
     */

    LOCAL_DECL int2 batch_offset[BATCH_SIZE];
    // Index for first grid point to update
    LOCAL_DECL int2 batch_min_uv[BATCH_SIZE];
    // Visibilities multiplied by weights
    LOCAL_DECL float2 batch_vis[NPOLS][BATCH_SIZE];

    // In-register sums
    Complex sums[MULTI_Y][MULTI_X][NPOLS];
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

    int u_phase = get_group_id(1) * TILE_X + get_local_id(0) * MULTI_X;
    int v_phase = get_group_id(2) * TILE_Y + get_local_id(1) * MULTI_Y;

    const int lid = get_local_id(1) * ${wgs_x} + get_local_id(0);
    int batch_start = get_group_id(0) * vis_per_workgroup;
    int batch_end = min(nvis, batch_start + vis_per_workgroup);
    for (; batch_start < batch_end; batch_start += BATCH_SIZE)
    {
        // Load batch
        int batch_size = min(batch_end - batch_start, BATCH_SIZE);
        if (lid < batch_size)
        {
            int vis_id = batch_start + lid;
            float sample_u = uvw[vis_id * 3];
            float sample_v = uvw[vis_id * 3 + 1];
            float sample_w = uvw[vis_id * 3 + 2];
            bool flipped = false;
            if (sample_w < 0) // TODO: eliminate this once uvw are preprocessed
            {
                sample_u = -sample_u;
                sample_v = -sample_v;
                sample_w = -sample_w;
                flipped = true;
            }
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
                if (flipped)
                    batch_vis[p][lid].y = -batch_vis[p][lid].y;
            }
            batch_min_uv[lid] = make_int2(min_u, min_v);
            int w_plane = __float2int_rn(min(sample_w, CONVOLVE_KERNEL_MAX_W) * CONVOLVE_KERNEL_W_SCALE);
            int offset_w = w_plane * CONVOLVE_KERNEL_W_STRIDE;
            batch_offset[lid] = make_int2(
                offset_w + sub_u * CONVOLVE_KERNEL_SLICE_STRIDE - min_u,
                offset_w + sub_v * CONVOLVE_KERNEL_SLICE_STRIDE - min_v);
        }

        BARRIER();

        // Process batch
        for (int vis_id = 0; vis_id < batch_size; vis_id++)
        {
            float2 sample_vis[NPOLS];
            for (int p = 0; p < NPOLS; p++)
                sample_vis[p] = batch_vis[p][vis_id];
            int2 min_uv = batch_min_uv[vis_id];
            int2 base_offset = batch_offset[vis_id];
            for (int y = 0; y < MULTI_Y; y++)
                for (int x = 0; x < MULTI_X; x++)
                {
                    // TODO: expand convolution kernel footprint such that
                    // multi-xy block is emitted as a unit, instead of separate
                    // cur_uv for each element.
                    int u = wrap(min_uv.x, CONVOLVE_KERNEL_SIZE_X, u_phase + x);
                    int v = wrap(min_uv.y, CONVOLVE_KERNEL_SIZE_Y, v_phase + y);
                    if (u != cur_uv[y][x].x || v != cur_uv[y][x].y)
                    {
                        writeback(out, out_row_stride, out_pol_stride, cur_uv[y][x].x, cur_uv[y][x].y, sums[y][x]);
                        cur_uv[y][x] = make_int2(u, v);
                        for (int p = 0; p < NPOLS; p++)
                            sums[y][x][p] = make_Complex(0.0f, 0.0f);
                    }
                    float2 weight_u = convolve_kernel[u + base_offset.x];
                    float2 weight_v = convolve_kernel[v + base_offset.y];
                    float2 weight = complex_mul(weight_u, weight_v);
                    for (int p = 0; p < NPOLS; p++)
                    {
                        // The weight is conjugated because the w kernel is
                        // for prediction rather than imaging.
                        sums[y][x][p] = Complex_madc(sample_vis[p], weight, sums[y][x][p]);
                    }
                }
        }

        // Necessary to prevent the next batch overwriting data for the current
        // one while still in progress
        BARRIER();
    }

    // Write back final internal values
    for (int y = 0; y < MULTI_Y; y++)
        for (int x = 0; x < MULTI_X; x++)
            writeback(out, out_row_stride, out_pol_stride, cur_uv[y][x].x, cur_uv[y][x].y, sums[y][x]);
}
