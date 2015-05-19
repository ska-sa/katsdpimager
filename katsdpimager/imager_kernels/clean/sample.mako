<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

<%def name="define_sample(real_type, size)">

typedef struct
{
    ${real_type} value;
    int2 pos;
} sample;

DEVICE_FN sample sample_max(sample a, sample b)
{
    return a.value > b.value ? a : b;
}

<%def name="sample_max(a, b, type)">sample_max((${a}), (${b}))</%def>
${wg_reduce.define_scratch('sample', size, 'reduce_sample_max_scratch')}
${wg_reduce.define_function('sample', size, 'reduce_sample_max', 'reduce_sample_max_scratch', sample_max)}

</%def> ## define_sample
