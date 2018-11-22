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
#define BIN_X ${bin_x}
#define BIN_Y ${bin_y}
#define CONVOLVE_KERNEL_SLICE_STRIDE ${convolve_kernel_slice_stride}
#define CONVOLVE_KERNEL_W_STRIDE ${convolve_kernel_w_stride}
#define make_Complex make_${real_type}2
#define atomic_accum_Complex_add atomic_accum_${real_type}2_add

/// Computes a * b
DEVICE_FN float2 Complex_mul(float2 a, float2 b)
{
    return make_float2(fma(a.x, b.x, -a.y * b.y),
                       fma(a.x, b.y, a.y * b.x));
}

/// Computes a * conj(b) + c
DEVICE_FN Complex Complex_madc(float2 a, float2 b, Complex c)
{
    Complex out;
    Complex a_ = make_Complex(a.x, a.y);
    Complex b_ = make_Complex(b.x, b.y);
    out.x = fma(a_.x, b_.x, fma(a_.y, b_.y, c.x));
    out.y = fma(a_.x, -b_.y, fma(a_.y, b_.x, c.y));
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
    int u0, int v0, const Complex values[MULTI_Y][MULTI_X][NPOLS])
{
    int offset = (v0 * out_row_stride + u0);
    for (int p = 0; p < NPOLS; p++)
    {
        for (int y = 0; y < MULTI_Y; y++)
            for (int x = 0; x < MULTI_X; x++)
                atomic_accum_Complex_add(&out[offset + y * out_row_stride + x], values[y][x][p]);
        offset += out_pol_stride;
    }
}

KERNEL REQD_WORK_GROUP_SIZE(${wgs_x}, ${wgs_y}, 1)
void grid(
    GLOBAL atomic_accum_Complex * RESTRICT out,
    int out_row_stride,
    int out_pol_stride,
    const GLOBAL float * RESTRICT weights,
    int weights_row_stride,
    int weights_pol_stride,
    const GLOBAL short4 * RESTRICT uv,
    const GLOBAL short * RESTRICT w_plane,
    const GLOBAL float2 * RESTRICT vis,
    const GLOBAL float2 * RESTRICT convolve_kernel,
    int uv_bias,
    int weights_address_bias,
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
    // Visibilities multiplied by all weights
    LOCAL_DECL float2 batch_vis[NPOLS][BATCH_SIZE];

    // In-register sums
    Complex sums[MULTI_Y][MULTI_X][NPOLS];
    // Last-known UV coordinates
    int cur_u0 = -1; // used as sentinel for first element in batch
    int cur_v0 = 0;

    int u_phase = get_group_id(0) * TILE_X + get_local_id(0) * MULTI_X;
    int v_phase = get_group_id(1) * TILE_Y + get_local_id(1) * MULTI_Y;

    const int lid = get_local_id(1) * ${wgs_x} + get_local_id(0);
    int batch_start = get_group_id(2) * vis_per_workgroup;
    int batch_end = min(nvis, batch_start + vis_per_workgroup);
    for (; batch_start < batch_end; batch_start += BATCH_SIZE)
    {
        // Load batch
        int batch_size = min(batch_end - batch_start, BATCH_SIZE);
        if (lid < batch_size)
        {
            int vis_id = batch_start + lid;
            short4 sample_uv = uv[vis_id];
            short sample_w_plane = w_plane[vis_id];
            int weight_address = sample_uv.y * weights_row_stride + sample_uv.x + weights_address_bias;
            for (int p = 0; p < NPOLS; p++, weight_address += weights_pol_stride)
            {
                // TODO: could improve this using float4s where appropriate
                int idx = vis_id * NPOLS + p;
                float sample_weight = weights[weight_address];
                float2 sample_vis = vis[idx];
                batch_vis[p][lid] = make_float2(sample_vis.x * sample_weight, sample_vis.y * sample_weight);
            }
            int2 min_uv = make_int2(sample_uv.x - uv_bias, sample_uv.y - uv_bias);
            int offset_w = sample_w_plane * CONVOLVE_KERNEL_W_STRIDE;
            batch_min_uv[lid] = min_uv;
            batch_offset[lid] = make_int2(
                offset_w + sample_uv.z * CONVOLVE_KERNEL_SLICE_STRIDE - min_uv.x,
                offset_w + sample_uv.w * CONVOLVE_KERNEL_SLICE_STRIDE - min_uv.y);
        }
        else
        {
            batch_min_uv[lid] = make_int2(0, 0);
            batch_offset[lid] = make_int2(0, 0);
            for (int p = 0; p < NPOLS; p++)
                batch_vis[p][lid] = make_float2(0.0f, 0.0f);
        }

        BARRIER();

        // Process batch
#pragma unroll 2
        for (int vis_id = 0; vis_id < BATCH_SIZE; vis_id++)
        {
            float2 sample_vis[NPOLS];
            float2 weight_u[MULTI_X];
            float2 weight_v[MULTI_Y];
            for (int p = 0; p < NPOLS; p++)
                sample_vis[p] = batch_vis[p][vis_id];
            int2 min_uv = batch_min_uv[vis_id];
            int2 base_offset = batch_offset[vis_id];
            int u0 = wrap(min_uv.x, BIN_X, u_phase);
            int v0 = wrap(min_uv.y, BIN_Y, v_phase);
            for (int x = 0; x < MULTI_X; x++)
                weight_u[x] = convolve_kernel[u0 + base_offset.x + x];
            for (int y = 0; y < MULTI_Y; y++)
                weight_v[y] = convolve_kernel[v0 + base_offset.y + y];
            if (u0 != cur_u0 || v0 != cur_v0)
            {
                if (cur_u0 >= 0)
                    writeback(out, out_row_stride, out_pol_stride, cur_u0, cur_v0, sums);
                cur_u0 = u0;
                cur_v0 = v0;
#if defined(__CUDA_ARCH__) || defined(__NV_CL_C_VERSION)
                /* CUDA 7 and 7.5 make an unfortunate de-optimisation choice:
                 * sums gets copied, then the original is zeroed, then the
                 * copies are atomically added to global memory. This adds an
                 * extra MULTI_X * MULTI_Y * N_POLS * 2 registers. Putting this
                 * empty asm statement here seems to prevent the
                 * de-optimisation.
                 */
                asm("");
#endif
                for (int y = 0; y < MULTI_Y; y++)
                    for (int x = 0; x < MULTI_X; x++)
                        for (int p = 0; p < NPOLS; p++)
                            sums[y][x][p] = make_Complex(0.0f, 0.0f);
            }
            for (int y = 0; y < MULTI_Y; y++)
                for (int x = 0; x < MULTI_X; x++)
                {
                    float2 weight = Complex_mul(weight_u[x], weight_v[y]);
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
    if (cur_u0 >= 0)
        writeback(out, out_row_stride, out_pol_stride, cur_u0, cur_v0, sums);
}
