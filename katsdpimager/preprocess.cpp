#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <boost/python.hpp>
#include <boost/make_shared.hpp>
#include <cstddef>
#include <stdint.h>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <cmath>

namespace py = boost::python;

namespace
{

template<int P>
struct vis_t
{
    int16_t uv[2];
    int16_t sub_uv[2];
    float weights[P];
    float vis[P][2];
    int16_t w_plane;
    int16_t w_slice;
    int32_t channel;
    int32_t baseline;
};

class visibility_collector_base
{
protected:
    float max_w;
    int w_slices;
    int w_planes;
    int oversample;
    py::object emit_callback;

public:
    std::int64_t num_input = 0, num_output = 0;

    visibility_collector_base(
        float max_w,
        int w_slices,
        int w_planes,
        int oversample,
        const py::object &emit_callback);
    virtual ~visibility_collector_base() {}
    virtual void add(
        int channel, int pixels, float cell_size,
        const py::object &uvw, const py::object &weights,
        const py::object &baselines, const py::object &vis) = 0;
    virtual void close() = 0;
};

visibility_collector_base::visibility_collector_base(
    float max_w,
    int w_slices,
    int w_planes,
    int oversample,
    const py::object &emit_callback)
    : max_w(max_w),
    w_slices(w_slices),
    w_planes(w_planes),
    oversample(oversample),
    emit_callback(emit_callback)
{
}

template<int P>
class visibility_collector : public visibility_collector_base
{
private:
    std::unique_ptr<vis_t<P>[]> buffer;
    std::size_t buffer_capacity;
    std::size_t buffer_size;
    py::object buffer_descr;

    void emit(vis_t<P> data[], std::size_t N);

    void add_contiguous(
        int channel, int pixels, float cell_size, std::size_t N,
        const float uvw[][3], const float weights[][P],
        const int32_t baselines[], const float vis[][P][2],
        vis_t<P> out[]);

    void compress();

public:
    visibility_collector(
        float max_w,
        int w_slices,
        int w_planes,
        int oversample,
        const py::object &emit_callback,
        std::size_t buffer_capacity);

    virtual void add(
        int channel, int pixels, float cell_size,
        const py::object &uvw, const py::object &weights,
        const py::object &baselines, const py::object &vis) override;

    virtual void close() override;
};

/* Enforce the array to have a specific type and shape. Only the built-in
 * types are supported. The returned object is guaranteed to
 * - be an array type
 * - be C contiguous and aligned
 * - have first dimension R unless R == -1
 * - be one-dimensional if C == -1, two-dimensional with C columns otherwise
 */
py::object get_array(const py::object &obj, int typenum, npy_intp R, npy_intp C)
{
    PyArray_Descr *descr = py::expect_non_null(PyArray_DescrFromType(typenum));
    PyObject *out = py::expect_non_null(PyArray_FromAny(
        obj.ptr(), descr, 0, 0,
        NPY_ARRAY_CARRAY_RO, NULL));
    py::object out_obj{py::handle<>(out)};
    int ndims = PyArray_NDIM((PyArrayObject *) out);
    if (ndims != (C == -1 ? 1 : 2))
    {
        PyErr_SetString(PyExc_TypeError, "Array has wrong number of dimensions");
        py::throw_error_already_set();
    }
    if (R != -1 && PyArray_DIMS((PyArrayObject *) out)[0] != R)
    {
        throw std::invalid_argument("Array has incorrect number of rows");
    }
    return out_obj;
}

template<typename T>
T *array_data(const py::object &array)
{
    return reinterpret_cast<T *>(PyArray_BYTES((PyArrayObject *) array.ptr()));
}

// TODO: optimise for oversample being a power of 2, in which case this is just a
// shift and mask.
void subpixel_coord(float x, int32_t oversample, int16_t &quo, int16_t &rem)
{
    int32_t xs = int32_t(floor(x * oversample));
    quo = xs / oversample;
    rem = xs % oversample;
    if (rem < 0)
    {
        quo--;
        rem += oversample;
    }
}

template<int P>
void visibility_collector<P>::add_contiguous(
    int channel, int pixels, float cell_size,
    std::size_t N,
    const float uvw[][3], const float weights[][P],
    const int32_t baselines[], const float vis[][P][2],
    vis_t<P> out[])
{
    float offset = pixels * 0.5f;
    float uv_scale = 1.0 / cell_size;
    float w_scale = (w_slices - 0.5f) * w_planes / max_w;
    int max_slice_plane = w_slices * w_planes - 1; // TODO: check for overflow?
    for (std::size_t row = 0; row < N; row++)
    {
        float u = uvw[row][0];
        float v = uvw[row][1];
        float w = uvw[row][2];
        if (w < 0.0f)
        {
            u = -u;
            v = -v;
            w = -w;
            for (int p = 0; p < P; p++)
            {
                out[row].vis[p][0] = vis[row][p][0];
                out[row].vis[p][1] = -vis[row][p][1]; // conjugate
            }
        }
        else
            std::memcpy(&out[row].vis, &vis[row], sizeof(vis[row]));
        for (int p = 0; p < P; p++)
        {
            float weight = weights[row][p];
            out[row].vis[p][0] *= weight;
            out[row].vis[p][1] *= weight;
        }
        u = u * uv_scale + offset;
        v = v * uv_scale + offset;
        // The plane number is biased by half a slice, because the first slice
        // is half-width and centered at w=0.
        w = trunc(w * w_scale + w_planes * 0.5f);
        int w_slice_plane = std::min(int(w), max_slice_plane);
        // TODO convert from here
        subpixel_coord(u, oversample, out[row].uv[0], out[row].sub_uv[0]);
        subpixel_coord(v, oversample, out[row].uv[1], out[row].sub_uv[1]);
        out[row].channel = channel;
        out[row].w_plane = w_slice_plane % w_planes;
        out[row].w_slice = w_slice_plane / w_planes;
        std::memcpy(&out[row].weights, &weights[row], sizeof(weights[row]));
        out[row].baseline = baselines[row];
    }
}

template<typename T>
struct compare
{
    bool operator()(const T &a, const T &b) const
    {
        if (a.channel != b.channel)
            return a.channel < b.channel;
        else if (a.w_slice != b.w_slice)
            return a.w_slice < b.w_slice;
        else
            return a.baseline < b.baseline;
    }
};

template<int P>
void visibility_collector<P>::emit(vis_t<P> data[], std::size_t N)
{
    npy_intp dims[1] = {(npy_intp) N};
    // PyArray_NewFromDescr steals a reference
    Py_INCREF(buffer_descr.ptr());
    PyObject *array = py::expect_non_null(PyArray_NewFromDescr(
        &PyArray_Type, (PyArray_Descr *) buffer_descr.ptr(), 1, dims, NULL, data,
        NPY_ARRAY_CARRAY, NULL));
    py::object obj{py::handle<>(array)};
    emit_callback(obj);
}

template<int P>
void visibility_collector<P>::compress()
{
    std::stable_sort(buffer.get(), buffer.get() + buffer_size, compare<vis_t<P> >());
    std::size_t out_pos = 0;
    vis_t<P> last;
    bool last_valid = false;
    for (std::size_t i = 0; i < buffer_size; i++)
    {
        const vis_t<P> &element = buffer[i];
        if (element.baseline < 0)
            continue;       // Autocorrelation
        if (last_valid
            && element.channel == last.channel
            && element.w_slice == last.w_slice
            && element.uv[0] == last.uv[0]
            && element.uv[1] == last.uv[1]
            && element.sub_uv[0] == last.sub_uv[0]
            && element.sub_uv[1] == last.sub_uv[1]
            && element.w_plane == last.w_plane)
        {
            for (int p = 0; p < P; p++)
                for (int j = 0; j < 2; j++)
                    last.vis[p][j] += element.vis[p][j];
            for (int p = 0; p < P; p++)
                last.weights[p] += element.weights[p];
        }
        else
        {
            if (last_valid)
            {
                buffer[out_pos] = last;
                out_pos++;
            }
            last = element;
            last_valid = true;
        }
    }
    if (last_valid)
    {
        buffer[out_pos] = last;
        out_pos++;
    }
    num_input += buffer_size;
    buffer_size = out_pos;

    // Identify ranges with the same channel and slice
    std::size_t prev = 0;
    int32_t last_channel = -1;
    int16_t last_w_slice = -1;
    for (std::size_t i = 0; i < buffer_size; i++)
    {
        if (buffer[i].channel != last_channel
            || buffer[i].w_slice != last_w_slice)
        {
            if (i != prev)
                emit(buffer.get() + prev, i - prev);
            prev = i;
            last_channel = buffer[i].channel;
            last_w_slice = buffer[i].w_slice;
        }
    }
    if (prev != buffer_size)
        emit(buffer.get() + prev, buffer_size - prev);
    num_output += buffer_size;
    buffer_size = 0;
}

template<int P>
void visibility_collector<P>::add(
    int channel, int pixels, float cell_size,
    const py::object &uvw_, const py::object &weights_,
    const py::object &baselines_, const py::object &vis_)
{
    py::object uvw = get_array(uvw_, NPY_FLOAT32, -1, 3);
    const std::size_t N = PyArray_DIMS((PyArrayObject *) uvw.ptr())[0];
    py::object weights = get_array(weights_, NPY_FLOAT32, N, P);
    py::object baselines = get_array(baselines_, NPY_INT32, N, -1);
    py::object vis = get_array(vis_, NPY_COMPLEX64, N, P);

    std::size_t start = 0;
    while (start < N)
    {
        std::size_t M = std::min(N - start, buffer_capacity - buffer_size);
        add_contiguous(channel, pixels, cell_size, M,
            array_data<const float[3]>(uvw) + start,
            array_data<const float[P]>(weights) + start,
            array_data<const int32_t>(baselines) + start,
            array_data<const float[P][2]>(vis) + start,
            buffer.get() + buffer_size);
        buffer_size += M;
        start += M;
        if (buffer_size == buffer_capacity)
            compress();
    }
}

template<int P>
void visibility_collector<P>::close()
{
    compress();
}

void dtype_add_field(
    py::list &names, py::list &formats, py::list &offsets,
    const char *name, int typenum, const py::tuple &dims, std::ptrdiff_t offset)
{
    PyArray_Descr *base_descr = py::expect_non_null(PyArray_DescrFromType(typenum));
    py::object base{py::handle<>((PyObject *) base_descr)};
    if (len(dims) == 0)
    {
        formats.append(base);
    }
    else
    {
        py::tuple subtype_args = py::make_tuple(base, dims);
        PyArray_Descr *subtype_descr;
        if (!PyArray_DescrAlignConverter(subtype_args.ptr(), &subtype_descr))
            py::throw_error_already_set();
        py::object subtype{py::handle<>((PyObject *) subtype_descr)};
        formats.append(subtype);
    }
    names.append(name);
    offsets.append(offset);
}

template<int P>
py::object make_vis_descr()
{
    py::list names, formats, offsets;
    py::dict dtype_dict;
    dtype_dict["names"] = names;
    dtype_dict["formats"] = formats;
    dtype_dict["offsets"] = offsets;
    dtype_dict["itemsize"] = sizeof(vis_t<P>);
#define ADD_FIELD(field, typenum, dims) \
        (dtype_add_field(names, formats, offsets, #field, typenum, py::make_tuple dims, \
                         offsetof(vis_t<P>, field)))
    ADD_FIELD(uv, NPY_INT16, (2));
    ADD_FIELD(sub_uv, NPY_INT16, (2));
    ADD_FIELD(weights, NPY_FLOAT32, (P));
    ADD_FIELD(vis, NPY_COMPLEX64, (P));
    ADD_FIELD(w_plane, NPY_INT16, ());
    ADD_FIELD(w_slice, NPY_INT16, ());
    ADD_FIELD(channel, NPY_INT32, ());
    ADD_FIELD(baseline, NPY_INT32, ());
#undef ADD_FIELD

    PyArray_Descr *descr;
    if (!PyArray_DescrConverter(dtype_dict.ptr(), &descr))
        py::throw_error_already_set();
    return py::object{py::handle<>((PyObject *) descr)};
}

template<int P>
visibility_collector<P>::visibility_collector(
    float max_w,
    int w_slices,
    int w_planes,
    int oversample,
    const py::object &emit_callback,
    std::size_t buffer_capacity)
    : visibility_collector_base(max_w, w_slices, w_planes, oversample, emit_callback),
    buffer(new vis_t<P>[buffer_capacity]),
    buffer_capacity(buffer_capacity),
    buffer_size(0),
    buffer_descr(make_vis_descr<P>())
{
}

// Factory for a visibility collector with *up to* P polarizations
template<int P>
boost::shared_ptr<visibility_collector_base>
make_visibility_collector(
    int polarizations,
    float max_w,
    int w_slices,
    int w_planes,
    int oversample,
    const py::object &emit_callback,
    std::size_t buffer_capacity)
{
    if (polarizations > P || polarizations <= 0)
        throw std::invalid_argument("polarizations must be 1, 2, 3 or 4");
    else if (polarizations == P)
        return boost::make_shared<visibility_collector<P> >(
            max_w, w_slices, w_planes, oversample,
            emit_callback, buffer_capacity);
    else
        // The special case for P=1 prevents an infinite template recursion.
        // When P=1, this code is unreachable.
        return make_visibility_collector<P == 1 ? 1 : P - 1>(
            polarizations, max_w, w_slices, w_planes, oversample,
            emit_callback, buffer_capacity);
}

/* Wrapper to deal with import_array returning nothing in Python 2, NULL in
 * Python 3.
 */
#if PY_MAJOR_VERSION >= 3
static void *call_import_array(bool &success)
#else
static void call_import_array(bool &success)
#endif
{
    success = false;
    import_array(); // This is a macro that might return
    success = true;
#if PY_MAJOR_VERSION >= 3
    return NULL;
#endif
}

} // anonymous namespace

BOOST_PYTHON_MODULE_INIT(_preprocess)
{
    using namespace boost::python;

    bool numpy_imported = false;
    call_import_array(numpy_imported);
    if (!numpy_imported)
        py::throw_error_already_set();
    class_<visibility_collector_base, boost::shared_ptr<visibility_collector_base>, boost::noncopyable>(
            "VisibilityCollector", no_init)
        .def("__init__", make_constructor(make_visibility_collector<4>))
        .def("add", &visibility_collector_base::add)
        .def("close", &visibility_collector_base::close)
        .def_readonly("num_input", &visibility_collector_base::num_input)
        .def_readonly("num_output", &visibility_collector_base::num_output)
    ;
}
