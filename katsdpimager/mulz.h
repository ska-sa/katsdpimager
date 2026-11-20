/**
 * @file
 *
 * MulZ template class.
 */

#include <limits>

/**
 * Adapter that behaves like the wrapped class, except that multiplication by
 * zero (on either side) always produces zero, even if the other value is
 * non-finite.
 */
template<typename T>
class MulZ
{
private:
    T value;

public:
    typedef T value_type;

    // In C++11, std::move is not constexpr, so we use static_cast instead
    constexpr MulZ(T value = T()) : value(static_cast<T &&>(value)) {}
    constexpr operator T() const { return value; }
    explicit constexpr operator bool() const { return value; }

    constexpr MulZ operator-() const { return MulZ(-value); }
    constexpr MulZ operator+(const MulZ &other) const { return MulZ(value + other.value); }
    constexpr MulZ operator-(const MulZ &other) const { return MulZ(value - other.value); }
    constexpr MulZ operator/(const MulZ &other) const { return MulZ(value / other.value); }
    MulZ &operator+=(const MulZ &other) { value += other.value; return *this; }
    MulZ &operator-=(const MulZ &other) { value -= other.value; return *this; }
    MulZ &operator/=(const MulZ &other) { value /= other.value; return *this; }

    constexpr MulZ operator *(const MulZ &other) const
    {
        return value != T(0) && other.value != T(0) ? value * other.value : T(0);
    }

    MulZ &operator *=(const MulZ &other)
    {
        value = value != T(0) && other.value != T(0) ? value * other.value : T(0);
        return *this;
    }

    constexpr bool operator==(const MulZ &other) const { return value == other.value; }
    constexpr bool operator!=(const MulZ &other) const { return value != other.value; }
};

static_assert(double(MulZ<double>(1.5) + MulZ<double>(2.5)) == 4.0, "MulZ::operator+ is broken");
static_assert(double(MulZ<double>(1.5) - MulZ<double>(2.5)) == -1.0, "MulZ::operator- is broken");
static_assert(double(MulZ<double>(1.5) * MulZ<double>(2.5)) == 3.75, "MulZ::operator* is broken");
static_assert(double(MulZ<double>(0.0) * MulZ<double>(2.5)) == 0.0, "MulZ::operator* is broken");
static_assert(double(MulZ<double>(0.0) * MulZ<double>(std::numeric_limits<double>::infinity())) == 0.0, "MulZ::operator* is broken");
static_assert(double(MulZ<double>(std::numeric_limits<double>::quiet_NaN()) * MulZ<double>(0.0)) == 0.0, "MulZ::operator* is broken");


// Eigen 3.4.0 (Ubuntu 22.04) requires explicit numeric traits for custom scalar
// types. Without this, Eigen cannot determine how to extract the real part from
// MulZ<std::complex<float>> causing a static_cast compilation error.
// This was not needed with Eigen 3.3.7 (Ubuntu 20.04) which was more permissive
// about implicit conversions.

#include <Eigen/Core>

namespace Eigen {
template<typename T>
struct NumTraits<MulZ<T>> : NumTraits<T>
{
    typedef MulZ<typename NumTraits<T>::Real> Real;
    typedef MulZ<typename NumTraits<T>::NonInteger> NonInteger;
    typedef MulZ<T> Nested;
    enum {
        IsComplex = NumTraits<T>::IsComplex,
        IsInteger = NumTraits<T>::IsInteger,
        IsSigned = NumTraits<T>::IsSigned,
        RequireInitialization = NumTraits<T>::RequireInitialization,
        ReadCost = NumTraits<T>::ReadCost,
        AddCost = NumTraits<T>::AddCost,
        MulCost = NumTraits<T>::MulCost
    };
};
} // namespace Eigen
