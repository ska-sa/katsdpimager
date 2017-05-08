#ifndef KATSDPIMAGER_OPTIONAL_H
#define KATSDPIMAGER_OPTIONAL_H

#include <type_traits>
#include <pybind11/pybind11.h>

// A very cut-down version of C++17 std::optional
template<typename T>
class optional
{
private:
    typename std::aligned_storage<sizeof(T), alignof(T)>::type storage;
    bool has_value_ = false;

    T *ptr() { return reinterpret_cast<T *>(&storage); }
    const T *ptr() const { return reinterpret_cast<const T *>(&storage); }
    template<typename U> void assign(U &&value);

public:
    typedef T value_type;

    optional() : has_value_(false) {}
    optional(const optional &other);
    optional(optional &&other);
    optional(const T &value);
    optional(T &&value);
    ~optional() { reset(); }

    const T *operator->() const { return ptr(); }
    T *operator->() { return ptr(); }
    const T &operator*() const { return *ptr(); }
    T &operator*() { return *ptr(); }

    optional &operator=(const optional &other);
    optional &operator=(optional &&other);
    optional &operator=(const T &value);
    optional &operator=(T &&value);

    void reset();

    explicit operator bool() const noexcept { return has_value_; }
};

template<typename T>
template<typename U>
void optional<T>::assign(U &&value)
{
    if (has_value_)
        *ptr() = std::forward<U>(value);
    else
    {
        new(ptr()) T(std::forward<U>(value));
        has_value_ = true;
    }
}

template<typename T>
optional<T>::optional(const optional &other)
{
    if (other)
        assign(*other);
}

template<typename T>
optional<T>::optional(optional &&other)
{
    if (other)
        assign(std::move(*other));
}

template<typename T>
optional<T>::optional(const T &value)
{
    assign(value);
}

template<typename T>
optional<T>::optional(T &&value)
{
    assign(std::move(value));
}

template<typename T>
optional<T> &optional<T>::operator=(const optional &other)
{
    if (other)
        assign(*other);
    else
        reset();
    return *this;
}

template<typename T>
optional<T> &optional<T>::operator=(optional &&other)
{
    if (other)
        assign(std::move(*other));
    else
        reset();
    return *this;
}

template<typename T>
optional<T> &optional<T>::operator=(const T &value)
{
    assign(value);
    return *this;
}

template<typename T>
optional<T> &optional<T>::operator=(T &&value)
{
    assign(std::move(value));
    return *this;
}

template<typename T>
void optional<T>::reset()
{
    if (has_value_)
    {
        ptr()->~T();
        has_value_ = false;
    }
}

namespace pybind11
{
namespace detail
{

// Based on pybind11's optional_caster for std::optional - but only
// supports conversion in Python -> C++ direction for now.
template<typename T>
struct type_caster<optional<T>>
{
private:
    using value_conv = make_caster<T>;

public:
    PYBIND11_TYPE_CASTER(optional<T>, _("Optional[") + value_conv::name() + _("]"));

    static handle cast(const T &src, return_value_policy policy, handle parent)
    {
        return false;
    }

    bool load(handle src, bool convert)
    {
        if (!src)
            return false;
        else if (src.is_none())
        {
            value.reset();
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

#endif // KATSDPIMAGER_OPTIONAL_H
