#include <cmath>
#include <initializer_list>
#include <limits>

#include "../src/guidance_limits.h"
#include "../src/imu_strapdown.h"

void require(bool condition)
{
    if (!condition)
        throw std::runtime_error("guidance or IMU math regression");
}

int main()
{
    require(bounded_target_axial_speed(2.5, -1, 3) == 1.5);
    require(bounded_target_axial_speed(5, -5, 3) == 0);
    require(bounded_target_axial_speed(5, 30, 3) == 0);
    require(bounded_target_axial_speed(5, -30, 3) == 0);
    require(bounded_target_axial_speed(2, -3.5, 3) == -1.5);
    require(terminal_speed_floor_cap(20, 1.5, 3, 120, 120, 0.02, 0.05) == 3);
    require(terminal_speed_floor_cap(20, 1.5, 3, 119, 120, 0.02, 0.05) == 1.5);
    require(terminal_speed_floor_cap(20, 1.5, 3, 120, 120, 0.06, 0.05) == 1.5);
    require(terminal_speed_floor_cap(2, 1.5, 3, 120, 120, 0.02, 0.05) == 2);
    bool invalid_terminal_floor = false;
    try {
        terminal_speed_floor_cap(20, 1.5, 3, 120, 0, 0.02, 0.05);
    } catch (const std::invalid_argument&) {
        invalid_terminal_floor = true;
    }
    require(invalid_terminal_floor);
    require(terminal_speed_floor_cap(20, 7, 12, 40, 40, -.15, .15) == 12);
    require(terminal_speed_floor_cap(20, 7, 12, 40, 40, -.16, .15) == 7);
    require(terminal_speed_floor_cap(20, 7, 0, 40, 40, 0, .15) == 7);
    require(terminal_speed_floor_cap(20, 15, 12, 40, 40, 0, .15) == 15);
    for (double capped : {0.0, .5, 2.0, 7.0, 12.0, 15.0, 20.0}) {
        require(terminal_speed_floor_cap(20, capped, 12, 300, 40, 0, .15) ==
                std::max(capped, 12.0));
        require(terminal_speed_floor_cap(20, capped, 0, 300, 40, 0, .15) == capped);
    }
    for (double invalid : {std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        for (int index = 0; index < 7; ++index) {
            double inputs[] = {20, 2, 12, 80, 40, .01, .15};
            inputs[index] = invalid;
            bool rejected = false;
            try {
                terminal_speed_floor_cap(inputs[0], inputs[1], inputs[2], inputs[3],
                                         inputs[4], inputs[5], inputs[6]);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected);
        }
    }
    for (int index : {0, 1, 2, 3, 4, 6}) {
        double inputs[] = {20, 2, 12, 80, 40, .01, .15};
        inputs[index] = -1;
        bool rejected = false;
        try {
            terminal_speed_floor_cap(inputs[0], inputs[1], inputs[2], inputs[3],
                                     inputs[4], inputs[5], inputs[6]);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected);
    }
    require(bounded_target_axial_speed(-1, -10, 3) == 0);
    require(bounded_target_axial_speed(5, 30, 0) == 0);
    bool invalid_speed_rejected = false;
    try {
        bounded_target_axial_speed(1, 1, -1);
    } catch (const std::invalid_argument&) {
        invalid_speed_rejected = true;
    }
    require(invalid_speed_rejected);
    double terminal_min = terminal_vision_range_minimum(1e9, 12, 18);
    require(terminal_min == 1e9);
    terminal_min = terminal_vision_range_minimum(terminal_min, 20.5, 19);
    require(terminal_min == 1e9); // Early scale reopening is not a terminal fly-through.
    terminal_min = terminal_vision_range_minimum(terminal_min, 8, 120);
    require(terminal_min == 8);
    require(terminal_vision_range_minimum(terminal_min, 18, 160) == 8);
    for (double invalid : {std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        for (int index = 0; index < 3; ++index) {
            invalid_speed_rejected = false;
            try {
                bounded_target_axial_speed(index == 0 ? invalid : 1,
                                           index == 1 ? invalid : -1,
                                           index == 2 ? invalid : 3);
            } catch (const std::invalid_argument&) {
                invalid_speed_rejected = true;
            }
            require(invalid_speed_rejected);
        }
    }
    require(braking_speed_limit(0, 6, .4) == 0);
    require(std::abs(braking_speed_limit(20, 10, 0) - 20) < 1e-9);
    double previous = 0;
    for (double distance : {.01, .1, 1.0, 5.0, 20.0, 45.0, 100.0}) {
        const double speed = braking_speed_limit(distance, 6, .4);
        require(speed > previous);
        require(speed * .4 + speed * speed / 12 <= distance + 1e-9);
        require(speed < braking_speed_limit(distance, 6, 0));
        require(speed > braking_speed_limit(distance, 6, .8));
        for (double delay : {.35, .5}) {
            const double limited = braking_speed_limit(distance, 6, delay);
            require(limited * delay + limited * limited / 12 <= distance + 1e-9);
        }
        previous = speed;
    }
    for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        for (int argument = 0; argument < 3; ++argument) {
            bool rejected = false;
            try {
                braking_speed_limit(argument == 0 ? invalid : 1,
                                    argument == 1 ? invalid : 6,
                                    argument == 2 ? invalid : .4);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected);

            require(vertical_closure_speed_limit(20, 70, 45) == 20.0 * 45 / 70);
            require(vertical_closure_speed_limit(20, 70, 45) ==
                    vertical_closure_speed_limit(20, 140, 90));
            require(vertical_closure_speed_limit(5, 70, 45) == 5.0 * 45 / 70);
            require(vertical_closure_speed_limit(5, 10, -4) == 2);
            require(vertical_closure_speed_limit(0, 10, 4) == 0);
            require(vertical_closure_speed_limit(20, 10, 0) == 0);
            require(std::isinf(vertical_closure_speed_limit(0, 0, 4)));
            for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::quiet_NaN()}) {
                rejected = false;
                try {
                    vertical_closure_speed_limit(invalid, 10, 4);
                } catch (const std::invalid_argument&) {
                    rejected = true;
                }
                require(rejected);
            }

            const Eigen::Quaterniond initial = Eigen::AngleAxisd(.7, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(.4, Eigen::Vector3d::UnitY());
            const Eigen::Vector3d body_rate(.2, .3, .1);
            Eigen::Quaterniond attitude = initial;
            Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
            const Eigen::Vector3d acceleration(3, -2, 1);
            const Eigen::Vector3d gravity(0, 0, 9.81);
            for (int i = 0; i < 200; ++i) {
                attitude = integrate_body_gyro(attitude, body_rate, .005);
                const Eigen::Quaterniond truth = initial * Eigen::Quaterniond(
                    Eigen::AngleAxisd(body_rate.norm() * (i + 1) * .005, body_rate.normalized()));
                const Eigen::Vector3d force = truth.conjugate() * (acceleration + gravity);
                velocity += (attitude * force - gravity) * .005;
            }
            const Eigen::Quaterniond expected = initial *
                Eigen::Quaterniond(Eigen::AngleAxisd(body_rate.norm(), body_rate.normalized()));
            require(attitude.angularDistance(expected) < 1e-10);
            require((velocity - acceleration).norm() < 1e-10);
            require(integrate_body_gyro(initial, Eigen::Vector3d::Zero(), .005)
                        .angularDistance(initial) < 1e-10);
            rejected = false;
            try {
                integrate_body_gyro(initial, body_rate, -1);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected);
        }
    }
    bool rejected = false;
    try {
        braking_speed_limit(1, 0, .4);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected);
}
