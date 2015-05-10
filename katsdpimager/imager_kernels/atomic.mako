## Wrappers for atomic operations. These do not yet support OpenCL but could
## easily be extended to do so.
##
## These operations have relaxed memory consistency, i.e. are not ordered with
## respect to any other operations.

#ifdef __OPENCL_VERSION__
# error "Atomic operations on OpenCL are not supported yet"
#else

% for T in ['int', 'uint', 'long', 'ulong', 'float', 'double']:
struct atomic_accum_${T}
{
    ${T} value;
};
% endfor

% for T in ['int', 'uint', 'float']:
DEVICE_FN void atomic_accum_${T}_add(atomic_accum_${T} *object, ${T} operand)
{
    atomicAdd(&object->value, operand);
}
% endfor

// CUDA doesn't support double precision atomics directly
DEVICE_FN void atomic_accum_double_add(atomic_accum_double *object, double operand)
{
    unsigned long long int *ptr = (unsigned long long int *) &object->value;
    unsigned long long int old = *ptr;
    unsigned long long int expected, updated;

    do {
        expected = old;
        updated = __double_as_longlong(operand + __longlong_as_double(old));
        old = atomicCAS(ptr, old, updated);
    } while (old != expected);
}

#endif // !__OPENCL_VERSION__

## Parts below here are portable

% for T in ['int', 'uint', 'float', 'double']:
% for V in [2, 4]:
typedef struct
{
    atomic_accum_${T} v[${V}];
} atomic_accum_${T}${V};

DEVICE_FN void atomic_accum_${T}${V}_add(atomic_accum_${T}${V} *object, ${T}${V} operand)
{
% for i in range(V):
    atomic_accum_${T}_add(&object->v[${i}], operand.${'xyzw'[i]});
% endfor
}
% endfor
% endfor
