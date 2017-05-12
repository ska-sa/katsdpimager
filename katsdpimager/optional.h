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

/* Based on pybind11's optional_caster for std::optional. It's supposed to be
 * possible to just derive from
 * pybind11::detail::optional_caster<boost::optional<T>>, but it doesn't
 * compile (https://github.com/pybind/pybind11/issues/847).
 */
template<typename T>
struct type_caster<boost::optional<T>>
{
private:
    using value_conv = make_caster<T>;

public:
    PYBIND11_TYPE_CASTER(boost::optional<T>, _("Optional[") + value_conv::name() + _("]"));

    static handle cast(const boost::optional<T> &src, return_value_policy policy, handle parent)
    {
        if (!src)
            return none().inc_ref();
        return value_conv::cast(*src, policy, parent);
    }

    bool load(handle src, bool convert)
    {
        if (!src)
            return false;
        else if (src.is_none())
        {
            value = boost::none;
            return true;
        }
        else
        {
            value_conv inner_caster;
            if (!inner_caster.load(src, convert))
                return false;
            value = cast_op<T>(inner_caster);
            return true;
        }
    }
};

} // namespace detail
} // namespace pybind11

using boost::optional;

#endif // Boost fallback

#endif // KATSDPIMAGER_OPTIONAL_H
