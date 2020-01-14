<%include file="/port.mako"/>

typedef ${real_type} Real;
typedef ${real_type}2 Real2;
#define make_Real2 make_${real_type}2

#define WGS ${wgs}
#define POLS ${num_polarizations}

KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1)
void predict(
    GLOBAL float2 * RESTRICT vis,
    const GLOBAL short4 * RESTRICT uv,
    const GLOBAL short * RESTRICT w_plane,
    const GLOBAL float * RESTRICT weights,
    const GLOBAL float * RESTRICT lmn,
    const GLOBAL float * RESTRICT flux,
    int n_vis,
    int n_sources,
    int oversample, float uv_scale,
    float w_scale, float w_bias
    )
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    bool overflow = gid >= n_vis;
    short4 local_uv = overflow ? make_short4(0, 0, 0, 0) : uv[gid];
    float u = (local_uv.x * oversample + local_uv.z + 0.5f) * uv_scale;
    float v = (local_uv.y * oversample + local_uv.w + 0.5f) * uv_scale;
    float w = overflow ? 0.0f : w_plane[gid] * w_scale + w_bias;
    Real2 ans[POLS];
    for (int i = 0; i < POLS; i++)
        ans[i] = make_Real2(0, 0);

    LOCAL_DECL float l[WGS];
    LOCAL_DECL float m[WGS];
    LOCAL_DECL float n[WGS];
    LOCAL_DECL Real b[POLS][WGS];

    for (int start = 0; start < n_sources; start += WGS)
    {
        int batch = min(WGS, n_sources - start);
        if (lid < batch)
        {
            int idx = start + lid;
            l[lid] = lmn[3 * idx];
            m[lid] = lmn[3 * idx + 1];
            n[lid] = lmn[3 * idx + 2];
            for (int i = 0; i < POLS; i++)
                b[i][lid] = flux[POLS * idx + i];
        }
        BARRIER();

        if (!overflow)
        {
            for (int i = 0; i < batch; i++)
            {
                float scaled_phase = l[i] * u + m[i] * v + n[i] * w;
                // Manual range reduction, because __sincosf only works well near zero
                scaled_phase -= roundf(scaled_phase);
                scaled_phase *= -2 * (float) M_PI;
                float2 K;
                __sincosf(scaled_phase, &K.y, &K.x);
                for (int j = 0; j < POLS; j++)
                {
                    Real pb = b[j][i];
                    ans[j].x += K.x * pb;
                    ans[j].y += K.y * pb;
                }
            }
        }
        BARRIER();
    }

    if (overflow)
        return;
    int out_idx = POLS * gid;
    for (int i = 0; i < POLS; i++)
    {
        float weight = weights[out_idx + i];
        float2 old = vis[out_idx + i];
        float2 float_ans;
        float_ans.x = old.x - (float) ans[i].x * weight;
        float_ans.y = old.y - (float) ans[i].y * weight;
        vis[out_idx + i] = float_ans;
    }
}
