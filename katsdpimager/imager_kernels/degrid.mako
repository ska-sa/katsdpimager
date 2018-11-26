<%include file="/port.mako"/>
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

<%wg_reduce:define_scratch type="${real_type}" size="${wgs_x * wgs_y}" scratch_type="scratch_t" allow_shuffle="${True}"/>
<%wg_reduce:define_function type="${real_type}" size="${wgs_x * wgs_y}" function="reduce_add" scratch_type="scratch_t" allow_shuffle="${True}" broadcast="${False}"/>

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
#define BIN_X ${bin_x}
#define BIN_Y ${bin_y}
#define CONVOLVE_KERNEL_SLICE_STRIDE ${convolve_kernel_slice_stride}
#define CONVOLVE_KERNEL_W_STRIDE ${convolve_kernel_w_stride}

/// Computes a * b
DEVICE_FN float2 complex_mul(float2 a, float2 b)
{
    return make_float2(fma(a.x, b.x, -a.y * b.y),
                       fma(a.x, b.y, a.y * b.x));
}

/// Computes a * b + c
DEVICE_FN Complex Complex_mad(float2 a, Complex b, Complex c)
{
    Complex a_ = make_Complex(a.x, a.y);
    Complex out;
    out.x = fma(a_.x, b.x, fma(a_.y, -b.y, c.x));
    out.y = fma(a_.x, b.y, fma(a_.y, b.x, c.y));
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
    int u0, int v0, Complex values[MULTI_Y][MULTI_X][NPOLS])
{
    int offset = (v0 * grid_row_stride + u0);
    for (int p = 0; p < NPOLS; p++)
    {
        for (int y = 0; y < MULTI_Y; y++)
            for (int x = 0; x < MULTI_X; x++)
                values[y][x][p] = grid[offset + y * grid_row_stride + x];
        offset += grid_pol_stride;
    }
}

typedef struct
{
    int2 batch_offset[BATCH_SIZE];
    // Index for first grid point to update
    int2 batch_min_uv[BATCH_SIZE];
    // Accumulated output visibilities
    Complex batch_vis[NPOLS][BATCH_SIZE];
    // Scratch area for reductions
    scratch_t scratch;
} subgroup_local;

KERNEL REQD_WORK_GROUP_SIZE(${wgs_x}, ${wgs_y}, ${wgs_z})
void degrid(
    const GLOBAL Complex* RESTRICT grid,
    int grid_row_stride,
    int grid_pol_stride,
    const GLOBAL short4 * RESTRICT uv,
    const GLOBAL short * RESTRICT w_plane,
    const GLOBAL float * RESTRICT weights,
    GLOBAL float2 * RESTRICT vis,
    const GLOBAL float2 * RESTRICT convolve_kernel,
    int uv_bias,
    int nvis)
{
    LOCAL_DECL subgroup_local lcl[SUBGROUPS];
    // Last-known UV coordinates
    int cur_u0 = -1;
    int cur_v0 = -1;
    // Cached grid data
    Complex cached_grid[MULTI_Y][MULTI_X][NPOLS];

    const int lid = get_local_id(1) * ${wgs_x} + get_local_id(0);
    const int subgroup = get_local_id(2);
    const int batch_id = get_group_id(0) * SUBGROUPS + subgroup;
    LOCAL subgroup_local *lclp = &lcl[subgroup];

    // Load batch
    int batch_start = batch_id * BATCH_SIZE;
    int vis_id = batch_start + lid;
    if (vis_id < nvis)
    {
        short4 sample_uv = uv[vis_id];
        short sample_w_plane = w_plane[vis_id];
        for (int p = 0; p < NPOLS; p++)
            lclp->batch_vis[p][lid] = make_Complex(0, 0);

        int2 min_uv = make_int2(sample_uv.x - uv_bias, sample_uv.y - uv_bias);
        int offset_w = sample_w_plane * CONVOLVE_KERNEL_W_STRIDE;
        lclp->batch_min_uv[lid] = min_uv;
        lclp->batch_offset[lid] = make_int2(
            offset_w + sample_uv.z * CONVOLVE_KERNEL_SLICE_STRIDE - min_uv.x,
            offset_w + sample_uv.w * CONVOLVE_KERNEL_SLICE_STRIDE - min_uv.y);
    }

    BARRIER();

    int batch_size = min(nvis - batch_start, BATCH_SIZE);
    // Process batch
    for (int tile_y = 0; tile_y < BIN_Y; tile_y += TILE_Y)
        for (int tile_x = 0; tile_x < BIN_X; tile_x += TILE_X)
        {
            const int u_phase = tile_x + get_local_id(0) * MULTI_X;
            const int v_phase = tile_y + get_local_id(1) * MULTI_Y;
            for (int vis_id = 0; vis_id < batch_size; vis_id++)
            {
                int2 min_uv = lclp->batch_min_uv[vis_id];
                int2 base_offset = lclp->batch_offset[vis_id];
                Complex contrib[NPOLS];
                float2 weight_u[MULTI_X];
                float2 weight_v[MULTI_Y];
                int u0 = wrap(min_uv.x, BIN_X, u_phase);
                int v0 = wrap(min_uv.y, BIN_Y, v_phase);
                for (int x = 0; x < MULTI_X; x++)
                    weight_u[x] = convolve_kernel[u0 + base_offset.x + x];
                for (int y = 0; y < MULTI_Y; y++)
                    weight_v[y] = convolve_kernel[v0 + base_offset.y + y];
                if (u0 != cur_u0 || v0 != cur_v0)
                {
                    load(grid, grid_row_stride, grid_pol_stride, u0, v0, cached_grid);
                    cur_u0 = u0;
                    cur_v0 = v0;
                }
                for (int y = 0; y < MULTI_Y; y++)
                    for (int x = 0; x < MULTI_X; x++)
                    {
                        float2 weight = complex_mul(weight_u[x], weight_v[y]);
                        for (int p = 0; p < NPOLS; p++)
                            contrib[p] = Complex_mad(weight, cached_grid[y][x][p],
                                                     (x || y) ? contrib[p] : make_Complex(0, 0));
                    }
                for (int p = 0; p < NPOLS; p++)
                {
                    // TODO: result could be kept in a register instead of
                    // shared memory.
                    Complex reduced = make_Complex(
                        reduce_add(contrib[p].x, lid, &lclp->scratch),
                        reduce_add(contrib[p].y, lid, &lclp->scratch));
                    if (lid == 0)
                    {
                        Complex old = lclp->batch_vis[p][vis_id];
                        reduced.x += old.x;
                        reduced.y += old.y;
                        lclp->batch_vis[p][vis_id] = reduced;
                    }
                }
            }
        }

    BARRIER();

    // Write back results from the batch
    // TODO: this isn't the optimal memory access order
    if (vis_id < nvis)
    {
        for (int p = 0; p < NPOLS; p++)
        {
            // TODO: could improve this using float4s where appropriate
            int idx = vis_id * NPOLS + p;
            float2 residual = vis[idx];
            float weight = weights[idx];
            float2 predicted;
% if real_type == 'float':
            predicted = lclp->batch_vis[p][lid];
% else:
            predicted = make_float2(
                lclp->batch_vis[p][lid].x,
                lclp->batch_vis[p][lid].y);
% endif
            residual.x -= predicted.x * weight;
            residual.y -= predicted.y * weight;
            vis[idx] = residual;
        }
    }
}
