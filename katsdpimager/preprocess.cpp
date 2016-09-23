/**
 * @file
 *
 * Implementation of the core preprocessing functions from preprocess.py.
 *
 * Most of the functionality is documented in the wrappers in preprocess.py.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <boost/python.hpp>
#include <boost/make_shared.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <complex>
#include <Eigen/Core>
#include "mulz.h"

namespace py = boost::python;

namespace
{

/**
 * Data for a visibility with @a P polarizations.
 */
template<int P>
struct vis_t
{
    std::int16_t uv[2];
    std::int16_t sub_uv[2];
    float weights[P];
    std::complex<float> vis[P];
    std::int16_t w_plane;    ///< Plane within W slice
    std::int16_t w_slice;    ///< W-stacking slice
    std::int32_t channel;
    std::int32_t baseline;   ///< Baseline ID
    std::uint32_t index;     ///< Original position, for sort stability
};

/**
 * Abstract base class. This is the type exposed to Python, but a subclass
 * specialized for the number of polarizations is actually instantiated.
 */
class visibility_collector_base
{
protected:
    // Gridding parameters
    float max_w;
    int w_slices;
    int w_planes;
    int oversample;
    /// Callback function called with compressed data as a numpy array
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

    /**
     * Add a batch of visibilities. The @c py::object parameters are all numpy
     * arrays. The parameter list indicates the preferred types, but they will
     * be cast if necessary (as well as converted to C-order contiguous).
     * Performance will be best if they are already in the appropriate type and
     * layout.
     *
     * Let P be the number of output polarizations, and Q be the number of
     * input polarizations.  When Q &lt; 4, it is possible that the input
     * polarizations are insufficient to compute the requested Stokes
     * parameters. It is the caller's responsibility to ensure this does not
     * happen.
     *
     * @param channel      Channel number
     * @param cell_size    Size of UV cells, in same units as @a uvw
     * @param uvw          UVW coordinates, Nx3, complex64
     * @param weights      Imaging weights, NxQ, float32
     * @param baselines    Baseline indices, N, int32 (negative for autocorrelations)
     * @param vis          Visibilities (in arbitrary frame), NxQ, complex64
     * @param feed_angle1, feed_angle2  Feed angles in radians for the two
     *                     antennas in the baseline, for rotating from
     *                     feed-relative to celestial frame, N, float32
     * @param mueller_stokes Mueller matrix that converts from celestial RL polarization
     *                     frame to the desired output Stokes parameters, Px4.
     * @param mueller_circular  Mueller matrix that converts from the frame
     *                     given in @a vis to a feed-relative RL (circular) polarization
     *                     frame, 4xQ.
     */
    virtual void add(
        int channel, float cell_size,
        const py::object &uvw, const py::object &weights,
        const py::object &baselines, const py::object &vis,
        const py::object &feed_angle1, const py::object &feed_angle2,
        const py::object &mueller_stokes,
        const py::object &mueller_circular) = 0;
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
    /// Storage for buffered visibilities
    std::unique_ptr<vis_t<P>[]> buffer;
    /// Allocated memory for @ref buffer
    std::size_t buffer_capacity;
    /// Number of valid entries in @ref buffer
    std::size_t buffer_size;
    /// numpy dtype for passing data in @ref buffer to Python
    py::object buffer_descr;

    /**
     * Wrapper around @ref visibility_collector_base::emit. It constructs the
     * numpy object to wrap the memory.
     */
    void emit(vis_t<P> data[], std::size_t N);

    /**
     * Sort and compress the buffer, and call emit for each contiguous
     * portion that belongs to the same w plane and channel.
     */
    void compress();

    template<int Q>
    void add_impl(
        int channel, float cell_size,
        const py::object &uvw, const py::object &weights,
        const py::object &baselines, const py::object &vis,
        const py::object &feed_angle1, const py::object &feed_angle2,
        const py::object &mueller_stokes,
        const py::object &mueller_circular);

public:
    visibility_collector(
        float max_w,
        int w_slices,
        int w_planes,
        int oversample,
        const py::object &emit_callback,
        std::size_t buffer_capacity);

    virtual void add(
        int channel, float cell_size,
        const py::object &uvw, const py::object &weights,
        const py::object &baselines, const py::object &vis,
        const py::object &feed_angle1, const py::object &feed_angle2,
        const py::object &mueller_stokes,
        const py::object &mueller_circular) override;

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

/**
 * Compute pixel and subpixel coordinates in grid. See grid.py for details.
 * TODO: optimise for oversample being a power of 2, in which case this is just
 * a shift and mask.
 */
void subpixel_coord(float x, std::int32_t oversample, std::int16_t &pixel, std::int16_t &subpixel)
{
    std::int32_t xs = std::int32_t(std::floor(x * oversample));
    pixel = xs / oversample;
    subpixel = xs % oversample;
    if (subpixel < 0)
    {
        pixel--;
        subpixel += oversample;
    }
}

/**
 * Sort comparison operator for visibilities. It sorts first by channel and w
 * slice, then by baseline, then by original position (i.e., making it
 * stable). This is done rather than using std::stable_sort, because the
 * std::stable_sort in libstdc++ uses a temporary memory allocation while
 * std::sort is in-place.
 */
template<typename T>
struct compare
{
    bool operator()(const T &a, const T &b) const
    {
        if (a.channel != b.channel)
            return a.channel < b.channel;
        else if (a.w_slice != b.w_slice)
            return a.w_slice < b.w_slice;
        else if (a.baseline != b.baseline)
            return a.baseline < b.baseline;
        else
            return a.index < b.index;
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
    num_output += N;
    emit_callback(obj);
}

template<int P>
void visibility_collector<P>::compress()
{
    if (buffer_size == 0)
        return;  // some code will break on an empty buffer
    std::sort(buffer.get(), buffer.get() + buffer_size, compare<vis_t<P> >());
    std::size_t out_pos = 0;
    // Currently accumulating visibility
    vis_t<P> last = buffer[0];
    for (std::size_t i = 1; i < buffer_size; i++)
    {
        const vis_t<P> &element = buffer[i];
        if (element.channel != last.channel
            || element.w_slice != last.w_slice)
        {
            // Moved to the next channel/slice, so pass what we have
            // back to Python
            buffer[out_pos++] = last;
            emit(buffer.get(), out_pos);
            out_pos = 0;
            last = element;
        }
        else if (element.uv[0] == last.uv[0]
            && element.uv[1] == last.uv[1]
            && element.sub_uv[0] == last.sub_uv[0]
            && element.sub_uv[1] == last.sub_uv[1]
            && element.w_plane == last.w_plane)
        {
            // Continue accumulating the current visibility
            for (int p = 0; p < P; p++)
                last.vis[p] += element.vis[p];
            for (int p = 0; p < P; p++)
                last.weights[p] += element.weights[p];
        }
        else
        {
            // Moved to the next output visibility
            buffer[out_pos++] = last;
            last = element;
        }
    }
    // Emit the final batch
    buffer[out_pos++] = last;
    emit(buffer.get(), out_pos);
    buffer_size = 0;
}

template<int P>
template<int Q>
void visibility_collector<P>::add_impl(
    int channel, float cell_size,
    const py::object &uvw_obj, const py::object &weights_obj,
    const py::object &baselines_obj, const py::object &vis_obj,
    const py::object &feed_angle1_obj, const py::object &feed_angle2_obj,
    const py::object &mueller_stokes_obj,
    const py::object &mueller_circular_obj)
{
    typedef Eigen::Matrix<std::complex<float>, P, Q> MatrixPQcf;
    typedef Eigen::Matrix<std::complex<float>, 4, Q> Matrix4Qcf;
    typedef Eigen::Matrix<std::complex<float>, P, 1> VectorPcf;
    typedef Eigen::Matrix<float, P, 1> VectorPf;

    // Coerce objects to proper arrays and validate types and dimensions
    py::object uvw_array = get_array(uvw_obj, NPY_FLOAT32, -1, 3);
    const std::size_t N = PyArray_DIMS((PyArrayObject *) uvw_array.ptr())[0];
    py::object weights_array = get_array(weights_obj, NPY_FLOAT32, N, Q);
    py::object baselines_array = get_array(baselines_obj, NPY_INT32, N, -1);
    py::object vis_array = get_array(vis_obj, NPY_COMPLEX64, N, Q);
    py::object feed_angle1_array = get_array(feed_angle1_obj, NPY_FLOAT32, N, -1);
    py::object feed_angle2_array = get_array(feed_angle2_obj, NPY_FLOAT32, N, -1);
    py::object mueller_circular_array = get_array(mueller_circular_obj, NPY_COMPLEX64, 4, Q);
    py::object mueller_stokes_array = get_array(mueller_stokes_obj, NPY_COMPLEX64, P, 4);

    /* Eigen doesn't allow column-major row vectors and vice versa (see
     * http://eigen.tuxfamily.org/bz/show_bug.cgi?id=416)
     * Note that the numpy arrays are C-order (row-major), but they're mapped
     * into Eigen as column-major, and hence are transposed.
     */
    constexpr auto data_major = (Q == 1) ? Eigen::RowMajor : Eigen::ColMajor;

    auto uvw = array_data<const float[3]>(uvw_array);
    auto baselines = array_data<const int32_t>(baselines_array);
    const Eigen::Map<Eigen::Matrix<MulZ<std::complex<float>>, Q, Eigen::Dynamic, data_major>> vis(
        array_data<MulZ<std::complex<float>>>(vis_array), Q, N);
    const Eigen::Map<Eigen::Matrix<MulZ<float>, Q, Eigen::Dynamic, data_major>> weights(
        array_data<MulZ<float>>(weights_array), Q, N);
    auto feed_angle1 = array_data<const float>(feed_angle1_array);
    auto feed_angle2 = array_data<const float>(feed_angle2_array);
    const Eigen::Map<Eigen::Matrix<std::complex<float>, 4, Q, (Q != 1 ? Eigen::RowMajor : Eigen::ColMajor)>> mueller_circular(
        array_data<std::complex<float>>(mueller_circular_array));
    const Eigen::Map<Eigen::Matrix<std::complex<float>, P, 4, Eigen::RowMajor>> mueller_stokes(
        array_data<std::complex<float>>(mueller_stokes_array));

    float uv_scale = 1.0f / cell_size;
    float w_scale = (w_slices - 0.5f) * w_planes / max_w;
    int max_slice_plane = w_slices * w_planes - 1; // TODO: check for overflow? precompute?
    for (std::size_t i = 0; i < N; i++)
    {
        if (baselines[i] < 0)
            continue; // autocorrelation
        if (buffer_size == buffer_capacity)
            compress();

        /* Convert to Stokes parameters. To correctly handle the weights we
         * need to compute the composite Mueller matrix. This is the product
         * of the following conversions:
         * - mueller_circular, which converts the input values to a circular
         *   frame (RR, RL, LR, LL).
         * - parallactic angle rotations (which are diagonal in circular frame)
         * - convert circular frame to Stokes parameters
         */
        Matrix4Qcf mueller = mueller_circular;
        std::complex<float> rotate1(std::cos(feed_angle1[i]), std::sin(feed_angle1[i]));
        std::complex<float> rotate2(std::cos(feed_angle2[i]), std::sin(feed_angle2[i]));
        std::complex<float> RRscale = rotate1 * conj(rotate2);
        std::complex<float> RLscale = rotate1 * rotate2;
        mueller.row(0) *= RRscale;
        mueller.row(1) *= RLscale;
        mueller.row(2) *= conj(RLscale);
        mueller.row(3) *= conj(RRscale);
        MatrixPQcf convert = mueller_stokes * mueller;
        VectorPcf xvis = (convert.template cast<MulZ<std::complex<float>>>() * vis.col(i))
                         .template cast<std::complex<float>>();

        /* Transform weights. Weights are proportional to inverse variance, so we
         * first convert to variance, then invert again once output variances are
         * known. Covariance is not modelled.
         *
         * The absolute values are taken first. Weights can't be negative, but
         * they can be -0.0, whose inverse is -Inf. If a mix of -Inf and +Inf
         * variances is allowed, then the combined variance could be NaN
         * rather than +Inf.
         */
        VectorPf xweights =
            (convert.cwiseAbs2().template cast<MulZ<float>>()
             * weights.col(i).cwiseAbs().cwiseInverse())
            .template cast<float>().cwiseInverse();

        vis_t<P> &out = buffer[buffer_size];
        float u = uvw[i][0];
        float v = uvw[i][1];
        float w = uvw[i][2];
        if (w < 0.0f)
        {
            u = -u;
            v = -v;
            w = -w;
            xvis = xvis.conjugate();
        }
        for (int p = 0; p < P; p++)
        {
            float weight = xweights(p);
            out.vis[p] = xvis(p) * weight;
            out.weights[p] = weight;
        }
        u = u * uv_scale;
        v = v * uv_scale;
        // The plane number is biased by half a slice, because the first slice
        // is half-width and centered at w=0.
        w = trunc(w * w_scale + w_planes * 0.5f);
        int w_slice_plane = std::min(int(w), max_slice_plane);
        // TODO convert from here
        subpixel_coord(u, oversample, out.uv[0], out.sub_uv[0]);
        subpixel_coord(v, oversample, out.uv[1], out.sub_uv[1]);
        out.channel = channel;
        out.w_plane = w_slice_plane % w_planes;
        out.w_slice = w_slice_plane / w_planes;
        out.baseline = baselines[i];
        // This could wrap if the buffer has > 4 billion elements, but that
        // can only affect efficiency, not correctness.
        out.index = (std::uint32_t) buffer_size;
        buffer_size++;
    }
    num_input += N;
}

template<int P>
void visibility_collector<P>::add(
    int channel, float cell_size,
    const py::object &uvw, const py::object &weights,
    const py::object &baselines, const py::object &vis,
    const py::object &feed_angle1, const py::object &feed_angle2,
    const py::object &mueller_stokes,
    const py::object &mueller_circular)
{
    PyObject *ptr = mueller_circular.ptr();
    if (!PyArray_Check(ptr))
        throw std::invalid_argument("mueller_circular is not an array");
    PyArrayObject *array = (PyArrayObject *) ptr;
    if (PyArray_NDIM(array) != 2)
        throw std::invalid_argument("mueller_circular is not 2D");
    int Q = PyArray_DIMS(array)[1];
    switch (Q)
    {
    case 1:
        add_impl<1>(channel, cell_size, uvw, weights, baselines, vis,
                    feed_angle1, feed_angle2,
                    mueller_stokes, mueller_circular);
        break;
    case 2:
        add_impl<2>(channel, cell_size, uvw, weights, baselines, vis,
                    feed_angle1, feed_angle2,
                    mueller_stokes, mueller_circular);
        break;
    case 3:
        add_impl<3>(channel, cell_size, uvw, weights, baselines, vis,
                    feed_angle1, feed_angle2,
                    mueller_stokes, mueller_circular);
        break;
    case 4:
        add_impl<4>(channel, cell_size, uvw, weights, baselines, vis,
                    feed_angle1, feed_angle2,
                    mueller_stokes, mueller_circular);
        break;
    default:
        throw std::invalid_argument("mueller_circular does not have 1-4 columns");
        break;
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
