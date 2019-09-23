#ifndef KATSDPIMAGER_OPTIONAL_H
#define KATSDPIMAGER_OPTIONAL_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if PYBIND11_HAS_OPTIONAL

using std::optional;

#elif PYBIND11_HAS_EXP_OPTIONAL

using std::experimental::optional;

#else

#include <boost/optional.hpp>
#include <boost/none.hpp>

namespace pybind11
{
namespace detail
{

template<typename T>
struct type_caster<boost::optional<T>> : optional_caster<boost::optional<T>> {};

} // namespace detail
} // namespace pybind11

using boost::optional;

#endif // Boost fallback

#endif // KATSDPIMAGER_OPTIONAL_H
