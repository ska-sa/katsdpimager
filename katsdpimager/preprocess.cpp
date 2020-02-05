/**
 * @file
 *
 * Implementation of the core preprocessing functions from preprocess.py.
 *
 * Most of the functionality is documented in the wrappers in preprocess.py.
 */

#define EIGEN_MPL2_ONLY 1    /* Avoid accidentally using non-MPL2 code */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <initializer_list>
#include <cstring>
#include <cmath>
#include <complex>
#include <memory>
#include <utility>
#include <type_traits>
#include <Eigen/Core>
#include "optional.h"
#include "mulz.h"

namespace py = pybind11;

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

struct channel_config
{
    float max_w;
    std::int32_t w_slices;
    std::int32_t w_planes;
    std::int32_t oversample;
    float cell_size;
};

namespace
{

// Recursive implementation for check_dimensions
static void check_dimensions_impl(const py::array &array, int axis)
{
}

template<typename T, typename ...Args>
static void check_dimensions_impl(const py::array &array, int axis, T&& dim, Args&&... dims)
{
    std::ptrdiff_t expected = std::forward<T>(dim);
    if (expected != -1 && expected != array.shape(axis))
        throw std::invalid_argument("Array has incorrect size");
    check_dimensions_impl(array, axis + 1, std::forward<Args>(dims)...);
}

/**
 * Ensures that an array has dimensions compatible with @a dims.
 *
 * Each element of @a dims is either a required size for the dimension, or
 * is -1 to indicate don't-care. The number of dimensions must also match.
 *
 * @exception std::invalid_argument if there is a mismatch
 */
template<typename ...T>
static void check_dimensions(const py::array &array, T&&... dims)
{
    if (array.ndim() != sizeof...(dims))
        throw std::invalid_argument("Array has incorrect number of dimensions");
    check_dimensions_impl(array, 0, std::forward<T>(dims)...);
}

static constexpr int array_flags = pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_ |
                                   pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_;

/**
 * Abstract base class. This is the type exposed to Python, but a subclass
 * specialized for the number of polarizations is actually instantiated.
 */
class visibility_collector_base
{
protected:
    // Gridding parameters
    py::array_t<channel_config, array_flags> config;
    /// Callback function called with compressed data as a numpy array
    std::function<void(py::array)> emit_callback;

public:
    std::int64_t num_input = 0, num_output = 0;

    std::size_t num_channels() const { return config.size(); }

    visibility_collector_base(
        py::array_t<channel_config, array_flags> config,
        std::function<void(py::array)> emit_callback);
    virtual ~visibility_collector_base() {}

    /**
     * Add a batch of visibilities. The @c py::object parameters are all numpy
     * arrays. The parameter list indicates the preferred types, but they will
     * be cast if necessary (as well as converted to C-order contiguous).
     * Performance will be best if they are already in the appropriate type and
     * layout.
     *
     * Let P be the number of output polarizations, Q the number of input
     * polarizations, and C the number of channels. When Q &lt; 4, it is
     * possible that the input polarizations are insufficient to compute the
     * requested Stokes parameters. It is the caller's responsibility to ensure
     * this does not happen.
     *
     * If no feed angle correction is needed, pass none for @a feed_angle1, @a
     * feed_angle2 and @a mueller_circular. In this case, @a mueller_stokes
     * converts directly from @a vis to the output Stokes parameters.
     *
     * @param uvw          UVW coordinates, Nx3, float32
     * @param weights      Imaging weights, CxNxQ, float32
     * @param baselines    Baseline indices, N, int32 (negative for autocorrelations)
     * @param vis          Visibilities (in arbitrary frame), CxNxQ, complex64
     * @param feed_angle1, feed_angle2  Feed angles in radians for the two
     *                     antennas in the baseline, for rotating from
     *                     celestial to feed-relative frame, N, float32
     * @param mueller_stokes Mueller matrix that converts from celestial RL polarization
     *                     frame to the desired output Stokes parameters, Px4.
     * @param mueller_circular  Mueller matrix that converts from the frame
     *                     given in @a vis to a feed-relative RL (circular) polarization
     *                     frame, 4xQ.
     */
    virtual void add(
        py::array_t<float, array_flags> uvw,
        py::array_t<float, array_flags> weights,
        py::array_t<std::int32_t, array_flags> baselines,
        py::array_t<std::complex<float>, array_flags> vis,
        optional<py::array_t<float, array_flags>> feed_angle1,
        optional<py::array_t<float, array_flags>> feed_angle2,
        py::array_t<std::complex<float>, array_flags> mueller_stokes,
        optional<py::array_t<std::complex<float>, array_flags>> mueller_circular) = 0;
    virtual void close() = 0;
};

visibility_collector_base::visibility_collector_base(
    py::array_t<channel_config, array_flags> config,
    std::function<void(py::array)> emit_callback)
    : config(std::move(config)), emit_callback(std::move(emit_callback))
{
    check_dimensions(this->config, -1);
}

/**
 * Static Mueller matrix, for the case of no parallactic angle correction.
 */
template<int P, int Q>
class mueller_generator_simple
{
public:
    typedef Eigen::Matrix<std::complex<float>, P, Q> result_type;

private:
    result_type mueller;

public:
    explicit mueller_generator_simple(const result_type &mueller) : mueller(mueller) {}
    const result_type &operator()(std::size_t) const { return mueller; }
};

/**
 * Mueller matrix computed with parallactic angle correction.
 */
template<int P, int Q>
class mueller_generator_parallactic
{
public:
    typedef Eigen::Matrix<std::complex<float>, P, Q> result_type;

private:
    const float *feed_angle1;
    const float *feed_angle2;
    const Eigen::Matrix<std::complex<float>, P, 4> stokes;
    const Eigen::Matrix<std::complex<float>, 4, Q> circular;

public:
    mueller_generator_parallactic(
        const float *feed_angle1, const float *feed_angle2,
        const Eigen::Matrix<std::complex<float>, P, 4> &stokes,
        const Eigen::Matrix<std::complex<float>, 4, Q> &circular);

    result_type operator()(std::size_t idx) const;
};

template<int P, int Q>
mueller_generator_parallactic<P, Q>::mueller_generator_parallactic(
    const float *feed_angle1, const float *feed_angle2,
    const Eigen::Matrix<std::complex<float>, P, 4> &stokes,
    const Eigen::Matrix<std::complex<float>, 4, Q> &circular)
    : feed_angle1(feed_angle1), feed_angle2(feed_angle2),
    stokes(stokes), circular(circular)
{
}

template<int P, int Q>
auto mueller_generator_parallactic<P, Q>::operator()(std::size_t idx) const -> result_type
{
    Eigen::Matrix<std::complex<float>, 4, Q> mueller = circular;
    std::complex<float> rotate1(std::cos(feed_angle1[idx]), std::sin(feed_angle1[idx]));
    std::complex<float> rotate2(std::cos(feed_angle2[idx]), std::sin(feed_angle2[idx]));
    std::complex<float> RRscale = rotate1 * conj(rotate2);
    std::complex<float> RLscale = rotate1 * rotate2;
    mueller.row(0) *= RRscale;
    mueller.row(1) *= RLscale;
    mueller.row(2) *= conj(RLscale);
    mueller.row(3) *= conj(RRscale);
    return stokes * mueller;
}

template<int P>
class visibility_collector : public visibility_collector_base
{
private:
    /// Storage for buffered visibilities
    py::array_t<vis_t<P>> buffer_storage;
    /// Pointer to start of @ref buffer_storage
    vis_t<P> *buffer;
    /// Allocated memory for @ref buffer
    std::size_t buffer_capacity;
    /// Number of valid entries in @ref buffer
    std::size_t buffer_size;

    /**
     * Wrapper around @ref visibility_collector_base::emit. It constructs the
     * numpy object to wrap the memory. @a data must be pointer into
     * @ref buffer_storage.
     */
    void emit(vis_t<P> data[], std::size_t N);

    /**
     * Sort and compress the buffer, and call emit for each contiguous
     * portion that belongs to the same w plane and channel.
     */
    void compress();

    template<int Q, typename Generator>
    void add_impl2(
        std::size_t N,
        const float uvw[][3], const float weights[],
        const std::int32_t baselines[], const std::complex<float> vis[],
        const Generator &gen);

    // Handles *up to* Q input polarizations
    template<int Q>
    void add_impl(
        py::array_t<float, array_flags> uvw,
        py::array_t<float, array_flags> weights,
        py::array_t<std::int32_t, array_flags> baselines,
        py::array_t<std::complex<float>, array_flags> vis,
        optional<py::array_t<float, array_flags>> feed_angle1,
        optional<py::array_t<float, array_flags>> feed_angle2,
        py::array_t<std::complex<float>, array_flags> mueller_stokes,
        optional<py::array_t<std::complex<float>, array_flags>> mueller_circular);

public:
    visibility_collector(
        py::array_t<channel_config, array_flags> config,
        std::function<void(py::array)> emit_callback,
        std::size_t buffer_capacity);

    virtual void add(
        py::array_t<float, array_flags> uvw,
        py::array_t<float, array_flags> weights,
        py::array_t<std::int32_t, array_flags> baselines,
        py::array_t<std::complex<float>, array_flags> vis,
        optional<py::array_t<float, array_flags>> feed_angle1,
        optional<py::array_t<float, array_flags>> feed_angle2,
        py::array_t<std::complex<float>, array_flags> mueller_stokes,
        optional<py::array_t<std::complex<float>, array_flags>> mueller_circular) override;

    virtual void close() override;
};

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
    py::array_t<vis_t<P>> array(N, data, buffer_storage);
    num_output += N;
    emit_callback(array);
}

template<int P>
void visibility_collector<P>::compress()
{
    if (buffer_size == 0)
        return;  // some code will break on an empty buffer
    std::sort(buffer, buffer + buffer_size, compare<vis_t<P> >());
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
            emit(buffer, out_pos);
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
    emit(buffer, out_pos);
    buffer_size = 0;
}

template<int P>
template<int Q, typename Generator>
void visibility_collector<P>::add_impl2(
    std::size_t N,
    const float uvw[][3], const float weights_raw[],
    const std::int32_t baselines[], const std::complex<float> vis_raw[],
    const Generator &gen)
{
    typedef Eigen::Matrix<std::complex<float>, P, Q> MatrixPQcf;
    typedef Eigen::Matrix<float, P, 1> VectorPf;
    typedef Eigen::Matrix<std::complex<float>, P, 1> VectorPcf;

    /* Eigen doesn't allow column-major row vectors and vice versa (see
     * http://eigen.tuxfamily.org/bz/show_bug.cgi?id=416)
     * Note that the numpy arrays are C-order (row-major), but they're mapped
     * into Eigen as column-major, and hence are transposed.
     */
    constexpr auto data_major = (Q == 1) ? Eigen::RowMajor : Eigen::ColMajor;

    for (std::size_t channel = 0; channel < num_channels(); channel++)
    {
        const Eigen::Map<Eigen::Matrix<MulZ<std::complex<float>>, Q, Eigen::Dynamic, data_major>> vis(
            reinterpret_cast<MulZ<std::complex<float>> *>(
                const_cast<std::complex<float> *>(vis_raw + channel * N * Q)), Q, N);
        const Eigen::Map<Eigen::Matrix<MulZ<float>, Q, Eigen::Dynamic, data_major>> weights(
            reinterpret_cast<MulZ<float> *>(
                const_cast<float *>(weights_raw + channel * N * Q)), Q, N);

        const channel_config &conf = config.at(channel);
        float uv_scale = 1.0f / conf.cell_size;
        float w_scale = (conf.w_slices - 0.5f) * conf.w_planes / conf.max_w;
        int max_slice_plane = conf.w_slices * conf.w_planes - 1; // TODO: check for overflow? precompute?
        for (std::size_t i = 0; i < N; i++)
        {
            if (baselines[i] < 0)
                continue;   // autocorrelation
            if (weights.col(i).cwiseEqual(0.0f).any())
            {
                /* Discard visibilities with zero weight on any polarisation.
                 * Zero weights generally indicates flagging, and if any
                 * polarisation if flagged then at least one of the inputs is
                 * contaminated and so no Stokes parameters will be clean.
                 */
                continue;
            }
            if (buffer_size == buffer_capacity)
                compress();

            // TODO: amortise this computation across channels
            const MatrixPQcf &convert = gen(i);
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
                std::complex<float> vis = xvis(p) * weight;
                if (!std::isfinite(vis.real()) || !std::isfinite(vis.imag()))
                {
                    // Squash visibilities with NaNs, which could come from
                    // calibration failures in katsdpcal
                    vis = 0.0f;
                    weight = 0.0f;
                }
                out.vis[p] = vis;
                out.weights[p] = weight;
            }
            u = u * uv_scale;
            v = v * uv_scale;
            // The plane number is biased by half a slice, because the first slice
            // is half-width and centered at w=0.
            w = trunc(w * w_scale + conf.w_planes * 0.5f);
            int w_slice_plane = std::min(int(w), max_slice_plane);
            // TODO convert from here
            subpixel_coord(u, conf.oversample, out.uv[0], out.sub_uv[0]);
            subpixel_coord(v, conf.oversample, out.uv[1], out.sub_uv[1]);
            out.channel = channel;
            out.w_plane = w_slice_plane % conf.w_planes;
            out.w_slice = w_slice_plane / conf.w_planes;
            out.baseline = baselines[i];
            // This could wrap if the buffer has > 4 billion elements, but that
            // can only affect efficiency, not correctness.
            out.index = (std::uint32_t) buffer_size;
            buffer_size++;
        }
        if (channel + 1 < num_channels())
        {
            // Not needed for correctness, but avoids putting the next channel
            // in the same buffer where it is unlikely to combine well.
            compress();
        }
    }
    num_input += num_channels() * N;
}

template<int P>
template<int Q>
void visibility_collector<P>::add_impl(
    py::array_t<float, array_flags> uvw_array,
    py::array_t<float, array_flags> weights_array,
    py::array_t<std::int32_t, array_flags> baselines_array,
    py::array_t<std::complex<float>, array_flags> vis_array,
    optional<py::array_t<float, array_flags>> feed_angle1_array,
    optional<py::array_t<float, array_flags>> feed_angle2_array,
    py::array_t<std::complex<float>, array_flags> mueller_stokes_array,
    optional<py::array_t<std::complex<float>, array_flags>> mueller_circular_array)
{
    if (vis_array.shape(2) < Q)
    {
        return add_impl<Q == 1 ? 1 : Q - 1>(
            std::move(uvw_array),
            std::move(weights_array),
            std::move(baselines_array),
            std::move(vis_array),
            std::move(feed_angle1_array),
            std::move(feed_angle2_array),
            std::move(mueller_stokes_array),
            std::move(mueller_circular_array));
    }
    check_dimensions(uvw_array, -1, 3);
    const std::size_t N = uvw_array.shape(0);
    const std::size_t C = num_channels();
    check_dimensions(weights_array, C, N, Q);
    check_dimensions(baselines_array, N);
    check_dimensions(vis_array, C, N, Q);

    const float (*uvw)[3] = reinterpret_cast<const float(*)[3]>(uvw_array.data());
    auto baselines = baselines_array.data();
    auto vis = vis_array.data();
    auto weights = weights_array.data();

    constexpr auto matrix_major = (Q != 1) ? Eigen::RowMajor : Eigen::ColMajor;

    if (!feed_angle1_array)
    {
        if (feed_angle2_array || mueller_circular_array)
            throw std::invalid_argument("feed_angle1 is None but feed_angle2 or mueller_circular is not");
        typedef Eigen::Ref<const Eigen::Matrix<std::complex<float>, P, Q, matrix_major>> mueller_stokes_t;
        auto mueller_stokes = mueller_stokes_array.cast<mueller_stokes_t>();
        mueller_generator_simple<P, Q> gen(mueller_stokes);
        add_impl2<Q, mueller_generator_simple<P, Q>>(
            N, uvw, weights, baselines, vis, gen);
    }
    else
    {
        if (!feed_angle2_array || !mueller_circular_array)
            throw std::invalid_argument("feed_angle1 is not None but feed_angle2 or mueller_circular is");
        check_dimensions(*feed_angle1_array, N);
        check_dimensions(*feed_angle2_array, N);
        auto feed_angle1 = feed_angle1_array->data();
        auto feed_angle2 = feed_angle2_array->data();
        typedef Eigen::Ref<const Eigen::Matrix<std::complex<float>, P, 4, Eigen::RowMajor>> mueller_stokes_t;
        typedef Eigen::Ref<const Eigen::Matrix<std::complex<float>, 4, Q, matrix_major>> mueller_circular_t;
        auto mueller_stokes = mueller_stokes_array.cast<mueller_stokes_t>();
        auto mueller_circular = mueller_circular_array->cast<mueller_circular_t>();
        mueller_generator_parallactic<P, Q> gen(feed_angle1, feed_angle2,
                                                mueller_stokes, mueller_circular);
        add_impl2<Q, mueller_generator_parallactic<P, Q>>(
            N, uvw, weights, baselines, vis, gen);
    }
}

template<int P>
void visibility_collector<P>::add(
    py::array_t<float, array_flags> uvw,
    py::array_t<float, array_flags> weights,
    py::array_t<std::int32_t, array_flags> baselines,
    py::array_t<std::complex<float>, array_flags> vis,
    optional<py::array_t<float, array_flags>> feed_angle1,
    optional<py::array_t<float, array_flags>> feed_angle2,
    py::array_t<std::complex<float>, array_flags> mueller_stokes,
    optional<py::array_t<std::complex<float>, array_flags>> mueller_circular)
{
    check_dimensions(vis, -1, -1, -1);
    std::size_t Q = vis.shape(2);
    if (Q < 1 || Q > 4)
        throw std::invalid_argument("only 4 input polarizations are supported");
    add_impl<4>(
        std::move(uvw),
        std::move(weights),
        std::move(baselines),
        std::move(vis),
        std::move(feed_angle1),
        std::move(feed_angle2),
        std::move(mueller_stokes),
        std::move(mueller_circular));
}

template<int P>
void visibility_collector<P>::close()
{
    compress();
}

template<int P>
visibility_collector<P>::visibility_collector(
    py::array_t<channel_config, array_flags> config,
    std::function<void(py::array)> emit_callback,
    std::size_t buffer_capacity)
    : visibility_collector_base(std::move(config), std::move(emit_callback)),
    buffer_storage(buffer_capacity),
    buffer(buffer_storage.mutable_data()),
    buffer_capacity(buffer_capacity),
    buffer_size(0)
{
}

// Factory for a visibility collector with *up to* P polarizations
template<int P>
static std::unique_ptr<visibility_collector_base>
make_visibility_collector(
    int polarizations,
    py::array_t<channel_config, array_flags> config,
    std::function<void(py::array)> emit_callback,
    std::size_t buffer_capacity)
{
    if (polarizations > P || polarizations <= 0)
        throw std::invalid_argument("polarizations must be 1, 2, 3 or 4");
    else if (polarizations == P)
    {
        visibility_collector_base *ptr = new visibility_collector<P>(
            std::move(config), std::move(emit_callback), buffer_capacity);
        return std::unique_ptr<visibility_collector_base>(ptr);
    }
    else
        // The special case for P=1 prevents an infinite template recursion.
        // When P=1, this code is unreachable.
        return make_visibility_collector<P == 1 ? 1 : P - 1>(
            polarizations, std::move(config),
            std::move(emit_callback), buffer_capacity);
}

// Registers those with P or fewer
template<int P>
static typename std::enable_if<P == 0>::type register_vis_dtypes() {}

template<int P>
static typename std::enable_if<0 < P>::type register_vis_dtypes()
{
    register_vis_dtypes<P - 1>();
    PYBIND11_NUMPY_DTYPE(vis_t<P>, uv, sub_uv, weights, vis, w_plane, w_slice, channel, baseline, index);
}

} // anonymous namespace

PYBIND11_MODULE(_preprocess, m)
{
    using namespace pybind11;

    m.doc() = "C++ backend of visibility preprocessing";
    register_vis_dtypes<4>();
    PYBIND11_NUMPY_DTYPE(channel_config, max_w, w_slices, w_planes, oversample, cell_size);

    class_<visibility_collector_base>(m, "VisibilityCollector")
        .def(init(&make_visibility_collector<4>))
        .def("add", &visibility_collector_base::add)
        .def("close", &visibility_collector_base::close)
        .def_readonly("num_input", &visibility_collector_base::num_input)
        .def_readonly("num_output", &visibility_collector_base::num_output)
    ;
    m.add_object("CHANNEL_CONFIG_DTYPE", py::dtype::of<channel_config>());
}
