#include <cmath>
#include <limits>
#include <stdexcept>

#include "../src/camera_pose_history.h"

void require(bool condition)
{
    if (!condition)
        throw std::runtime_error("camera pose timestamp regression");
}

int main()
{
    CameraPoseHistory history;
    const Eigen::Quaterniond first = Eigen::Quaterniond::Identity();
    const Eigen::Quaterniond second(Eigen::AngleAxisd(.6, Eigen::Vector3d::UnitY()));
    require(!history.at(1));
    require(history.add(1, first));
    require(history.add(1.1, second));
    require(!history.add(1.1, first));
    require(!history.add(.9, first));
    require(!history.add(2, Eigen::Quaterniond(0, 0, 0, 0)));
    require(!history.add(std::numeric_limits<double>::quiet_NaN(), first));
    require(history.at(1)->angularDistance(first) < 1e-10);
    require(history.at(1.1)->angularDistance(second) < 1e-10);
    require(history.at(1.05)->angularDistance(first.slerp(.5, second)) < 1e-10);
    require(!history.at(.99));
    require(!history.at(1.11));
    require(!history.at(std::numeric_limits<double>::quiet_NaN()));

    const Eigen::Vector3d fixed_target(100, 0, 20);
    const Eigen::Vector3d measured_ray = first.conjugate() * fixed_target.normalized();
    require(((*history.at(1)) * measured_ray - fixed_target.normalized()).norm() < 1e-10);
    require((second * measured_ray - fixed_target.normalized()).norm() > .5);

    require(history.add(2, first));
    require(!history.at(1.5));
    for (int i = 1; i <= 100; ++i)
        require(history.add(2 + .01 * i, first));
    require(!history.at(2));
    require(history.at(3).has_value());
}
