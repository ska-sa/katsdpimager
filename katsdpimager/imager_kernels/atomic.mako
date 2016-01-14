## Wrappers for atomic operations. These do not yet support OpenCL but could
## easily be extended to do so.
##
## These operations have relaxed memory consistency, i.e. are not ordered with
## respect to any other operations.
##
## Defining KATSDPIMAGER_DISABLE_ATOMICS causes the operations to be done with
## regular operations instead of atomics. This is obviously not safe, and is
## provided as a debugging/profiling tool to measure the effect of atomics on
## performance.

## For cases where floating-point atomics are not available, simulates them
## using integer atomic compare-and-swap.
<%def name="float_wrapper(float_t, int_t, cas, float_as_int, int_as_float, volatile)">
DEVICE_FN void atomic_accum_${float_t}_add(GLOBAL atomic_accum_${float_t} *object, ${float_t} operand)
{
    GLOBAL ${volatile} ${int_t} *ptr = (GLOBAL ${volatile} ${int_t} *) &object->value;
    ${int_t} old = *ptr;
    ${int_t} expected, updated;

    do {
        expected = old;
        updated = ${float_as_int}(operand + ${int_as_float}(old));
        old = ${cas}(ptr, old, updated);
    } while (old != expected);
}
</%def>

#ifdef KATSDPIMAGER_DISABLE_ATOMICS

% for T in ['int', 'uint', 'float', 'double']:
typedef struct
{
    ${T} value;
} atomic_accum_${T};

DEVICE_FN void atomic_accum_${T}_add(GLOBAL atomic_accum_${T} *object, ${T} operand)
{
    object->value += operand;
}
%endfor

#elif defined(__OPENCL_VERSION__)

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

% for T in ['int', 'uint', 'float', 'double']:
typedef struct
{
    volatile ${T} value;
} atomic_accum_${T};
%endfor

% for T in ['int', 'uint']:
DEVICE_FN void atomic_accum_${T}_add(GLOBAL atomic_accum_${T} *object, ${T} operand)
{
    atomic_add(&object->value, operand);
}
% endfor

// OpenCL doesn't support non-integer atomics directly
${float_wrapper('float', 'unsigned int', 'atomic_cmpxchg', 'as_uint', 'as_float', 'volatile')}
${float_wrapper('double', 'unsigned long', 'atomic_cmpxchg', 'as_ulong', 'as_double', 'volatile')}

#else  // !__OPENCL_VERSION__

% for T in ['int', 'uint', 'float', 'double']:
struct atomic_accum_${T}
{
    ${T} value;
};
% endfor

% for T in ['int', 'uint', 'float']:
DEVICE_FN void atomic_accum_${T}_add(GLOBAL atomic_accum_${T} *object, ${T} operand)
{
    atomicAdd(&object->value, operand);
}
% endfor

// CUDA doesn't support double precision atomics directly
${float_wrapper('double', 'unsigned long long', 'atomicCAS', '__double_as_longlong', '__longlong_as_double', '')}

#endif // !__OPENCL_VERSION__

## Parts below here are portable

% for T in ['int', 'uint', 'float', 'double']:
% for V in [2, 4]:
typedef struct
{
    atomic_accum_${T} v[${V}];
} atomic_accum_${T}${V};

DEVICE_FN void atomic_accum_${T}${V}_add(GLOBAL atomic_accum_${T}${V} *object, ${T}${V} operand)
{
% for i in range(V):
    atomic_accum_${T}_add(&object->v[${i}], operand.${'xyzw'[i]});
% endfor
}
% endfor
% endfor
