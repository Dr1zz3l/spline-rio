#pragma once
#include <Eigen/Core>

// Minimal sym::Rot3 shim for SymForce-generated C++ headers.
// Only implements .Data() which is the sole API used by generated code.

namespace sym {

template <typename Scalar>
class Rot3 {
public:
    using DataVec = Eigen::Matrix<Scalar, 4, 1>;
    explicit Rot3(const DataVec& qxyzw) : data_(qxyzw) {}
    const DataVec& Data() const { return data_; }

private:
    DataVec data_;
};

using Rot3d = Rot3<double>;

}  // namespace sym
