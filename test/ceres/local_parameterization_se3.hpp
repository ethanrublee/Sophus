#ifndef SOPHUS_TEST_LOCAL_PARAMETERIZATION_SE3_HPP
#define SOPHUS_TEST_LOCAL_PARAMETERIZATION_SE3_HPP

#include <ceres/local_parameterization.h>
#include <sophus/se3.hpp>

#define USE_CERES_AUTO_DIFFERENTIATION
#ifdef USE_CERES_AUTO_DIFFERENTIATION
#include "ceres/internal/autodiff.h"
#endif

namespace Sophus {
namespace test {

struct SE3Plus {
  template <typename Scalar>
  bool operator()(const Scalar* T_raw, const Scalar* delta_raw,
                  Scalar* T_plus_delta_raw) const {
    Eigen::Map<SE3<Scalar> const> const T(T_raw);
    Eigen::Map<Eigen::Matrix<Scalar, 6, 1> const> const delta(delta_raw);
    Eigen::Map<SE3<Scalar> > T_plus_delta(T_plus_delta_raw);
    T_plus_delta = T * SE3<Scalar>::exp(delta);
    return true;
  }
};

struct SE3PlusAt {
  SE3PlusAt(SE3d T) : T_(T) {}
  template <typename Scalar>
  bool operator()(const Scalar* delta_raw, Scalar* T_plus_delta_raw) const {
    SE3<Scalar> T = T_.cast<Scalar>();
    return SE3Plus()(T.data(), delta_raw, T_plus_delta_raw);
  }
  SE3d T_;
};
class LocalParameterizationSE3 : public ceres::LocalParameterization {
 public:
  virtual ~LocalParameterizationSE3() {}

  // SE3 plus operation for Ceres
  //
  //  T * exp(x)
  //
  virtual bool Plus(double const* T_raw, double const* delta_raw,
                    double* T_plus_delta_raw) const {
    return SE3Plus()(T_raw, delta_raw, T_plus_delta_raw);
  }

  // Jacobian of SE3 plus operation for Ceres
  //
  // Dx T * exp(x)  with  x=0
  //
  virtual bool ComputeJacobian(double const* T_raw,
                               double* jacobian_raw) const {
#ifdef USE_CERES_AUTO_DIFFERENTIATION
    using ParameterDims =
        typename ceres::SizedCostFunction<7, 6>::ParameterDims;

    std::array<double, 6> zero({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    std::array<const double*, 1> parameters({zero.data()});
    constexpr int kOutput = 7;
    double output[kOutput];

    ceres::internal::AutoDifferentiate<kOutput, ParameterDims>(
        SE3PlusAt(Eigen::Map<SE3d const>(T_raw)), parameters.data(), kOutput,
        output, &jacobian_raw);
    return true;
#else
    Eigen::Map<SE3d const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > jacobian(
        jacobian_raw);
    jacobian = T.Dx_this_mul_exp_x_at_0();
    return true;
#endif
  }

  virtual int GlobalSize() const { return SE3d::num_parameters; }

  virtual int LocalSize() const { return SE3d::DoF; }
};
}  // namespace test
}  // namespace Sophus

#endif
