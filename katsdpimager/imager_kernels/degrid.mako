<%include file="/port.mako"/>
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

<%wg_reduce:define_scratch type="${real_type}" size="${wgs_x * wgs_y}" scratch_type="scratch_t"/>
<%wg_reduce:define_function type="${real_type}" size="${wgs_x * wgs_y}" function="reduce_add" scratch_type="scratch_t"/>

typedef ${real_type} Real;
typedef ${real_type}2 Complex;
#define make_Complex make_${real_type}2
#define BATCH_SIZE ${wgs_x * wgs_y}
#define NPOLS ${num_polarizations}
#define TILE_X ${multi_x * wgs_x}
#define TILE_Y ${multi_y * wgs_y}
#define MULTI_X ${multi_x}
#define MULTI_Y ${multi_y}
#define SUBGROUPS ${wgs_z}
#define CONVOLVE_KERNEL_SIZE_X ${convolve_kernel_size_x}
#define CONVOLVE_KERNEL_SIZE_Y ${convolve_kernel_size_y}
#define CONVOLVE_KERNEL_OVERSAMPLE ${convolve_kernel_oversample}
#define CONVOLVE_KERNEL_OVERSAMPLE_MASK ${convolve_kernel_oversample - 1}
#define CONVOLVE_KERNEL_OVERSAMPLE_SHIFT ${int.bit_length(convolve_kernel_oversample) - 1}
#define CONVOLVE_KERNEL_SLICE_STRIDE ${convolve_kernel_slice_stride}
#define CONVOLVE_KERNEL_W_STRIDE ${convolve_kernel_w_stride}
#define CONVOLVE_KERNEL_W_SCALE ${convolve_kernel_w_scale}f
#define CONVOLVE_KERNEL_MAX_W ${convolve_kernel_max_w}f

/// Computes a * b
DEVICE_FN float2 complex_mul(float2 a, float2 b)
{
    return make_float2(fma(a.x, b.x, -a.y * b.y),
                       fma(a.x, b.y, a.y * b.x));
}

/// Computes a * b + c
DEVICE_FN Complex Complex_mad(float2 a, Complex b, Complex c)
{
    Complex out;
    out.x = fma(a.x, b.x, fma(a.y, -b.y, c.x));
    out.y = fma(a.x, b.y, fma(a.y, b.x, c.y));
    return out;
}

/**
 * Finds smallest value such that ans % tile_size == pos and
 * ans >= low. It requires that tile_size be a power of 2.
 *
 * @todo Share code with grid.mako.
 */
DEVICE_FN int wrap(int low, int tile_size, int pos)
{
    return ((low - pos + tile_size - 1) & ~(tile_size - 1)) + pos;
}

DEVICE_FN void load(
    const GLOBAL Complex * RESTRICT grid,
    int grid_row_stride,
    int grid_pol_stride,
    int u, int v, Complex values[NPOLS])
{
    int offset = (v * grid_row_stride + u);
    for (int p = 0; p < NPOLS; p++)
    {
        values[p] = grid[offset];
        offset += grid_pol_stride;
    }
}

KERNEL REQD_WORK_GROUP_SIZE(${wgs_x}, ${wgs_y}, ${wgs_z})
void degrid(
    const GLOBAL Complex* RESTRICT grid,
    int grid_row_stride,
    int grid_pol_stride,
    const GLOBAL short4 * RESTRICT uv,
    const GLOBAL short * RESTRICT w_plane,
    GLOBAL float2 * RESTRICT vis,
    const GLOBAL float2 * RESTRICT convolve_kernel,
    int vis_per_subgroup,
    int nvis)
{
    LOCAL_DECL int2 batch_offset[SUBGROUPS][BATCH_SIZE];
    // Index for first grid point to update
    LOCAL_DECL int2 batch_min_uv[SUBGROUPS][BATCH_SIZE];
    // Accumulated output visibilities
    LOCAL_DECL Complex batch_vis[SUBGROUPS][NPOLS][BATCH_SIZE];
    // Scratch area for reductions
    LOCAL_DECL scratch_t scratch[SUBGROUPS];
    // Last-known UV coordinates
    int2 cur_uv[MULTI_Y][MULTI_X];
    // Cached grid data
    Complex cached_grid[MULTI_Y][MULTI_X][NPOLS];

    const int lid = get_local_id(1) * ${wgs_x} + get_local_id(0);
    const int subgroup = get_local_id(2);
    const int batch_id = get_group_id(0) * SUBGROUPS + subgroup;

    // Initialize things
    for (int y = 0; y < MULTI_Y; y++)
        for (int x = 0; x < MULTI_X; x++)
        {
            cur_uv[y][x] = make_int2(-1, -1);
        }

    int batch_start = batch_id * vis_per_subgroup;
    int batch_end = min(nvis, batch_start + vis_per_subgroup);
    // TODO: restructure this whole loop so that local memory addresses
    // aren't split up into subgroup/vis_id.
    for (; batch_start < batch_end; batch_start += BATCH_SIZE)
    {
        // Load batch
        int batch_size = min(batch_end - batch_start, BATCH_SIZE);
        if (lid < batch_size)
        {
            int vis_id = batch_start + lid;
            short4 sample_uv = uv[vis_id];
            short sample_w_plane = w_plane[vis_id];
            for (int p = 0; p < NPOLS; p++)
                batch_vis[subgroup][p][lid] = make_Complex(0, 0);

            int2 min_uv = make_int2(sample_uv.x, sample_uv.y);
            int offset_w = sample_w_plane * CONVOLVE_KERNEL_W_STRIDE;
            batch_min_uv[subgroup][lid] = min_uv;
            batch_offset[subgroup][lid] = make_int2(
                offset_w + sample_uv.z * CONVOLVE_KERNEL_SLICE_STRIDE - min_uv.x,
                offset_w + sample_uv.w * CONVOLVE_KERNEL_SLICE_STRIDE - min_uv.y);
        }

        BARRIER();

        // Process batch
        for (int tile_y = 0; tile_y < CONVOLVE_KERNEL_SIZE_Y; tile_y += TILE_Y)
            for (int tile_x = 0; tile_x < CONVOLVE_KERNEL_SIZE_X; tile_x += TILE_X)
            {
                const int u_phase = tile_x + get_local_id(0) * MULTI_X;
                const int v_phase = tile_y + get_local_id(1) * MULTI_Y;
                for (int vis_id = 0; vis_id < batch_size; vis_id++)
                {
                    int2 min_uv = batch_min_uv[subgroup][vis_id];
                    int2 base_offset = batch_offset[subgroup][vis_id];
                    Complex contrib[NPOLS];
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
                                cur_uv[y][x] = make_int2(u, v);
                                load(grid, grid_row_stride, grid_pol_stride, u, v, cached_grid[y][x]);
                            }
                            float2 weight_u = convolve_kernel[u + base_offset.x];
                            float2 weight_v = convolve_kernel[v + base_offset.y];
                            float2 weight = complex_mul(weight_u, weight_v);
                            for (int p = 0; p < NPOLS; p++)
                                contrib[p] = Complex_mad(weight, cached_grid[y][x][p],
                                                         (x || y) ? contrib[p] : make_Complex(0, 0));
                        }
                    for (int p = 0; p < NPOLS; p++)
                    {
                        // TODO: replace with Kepler shuffle where possible
                        // TODO: result could be kept in a register instead of
                        // shared memory.
                        Complex reduced = make_Complex(
                            reduce_add(contrib[p].x, lid, &scratch[subgroup]),
                            reduce_add(contrib[p].y, lid, &scratch[subgroup]));
                        if (lid == 0)
                        {
                            Complex old = batch_vis[subgroup][p][vis_id];
                            reduced.x += old.x;
                            reduced.y += old.y;
                            batch_vis[subgroup][p][vis_id] = reduced;
                        }
                    }
                }
            }

        BARRIER();

        // Write back results from the batch
        // TODO: this isn't the optimal memory access order
        if (lid < batch_size)
        {
            int vis_id = batch_start + lid;
            for (int p = 0; p < NPOLS; p++)
            {
                // TODO: could improve this using float4s where appropriate
                int idx = vis_id * NPOLS + p;
% if real_type == 'float':
                vis[idx] = batch_vis[subgroup][p][lid];
% else:
                vis[idx] = make_float2(
                    batch_vis[subgroup][p][lid].x,
                    batch_vis[subgroup][p][lid].y);
% endif
            }
        }

        /* No barrier necessary here: although the start of the next loop
         * zero-initializes batch_vis, it is done with the same work items
         * that read out the values here, and so there are no cross-thread
         * race conditions.
         */
    }
}
