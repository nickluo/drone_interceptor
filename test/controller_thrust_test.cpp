#include "../src/advanced_hybrid_controller.h"

namespace
{
    void require(bool condition, const char *message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    AdvancedHybridController controller(bool fov = true)
    {
        const double half_v = std::atan((960.0 / 1536.0) * std::tan(36.0 * M_PI / 180.0));
        AdvancedHybridController c(3.5, 5.0, 5.0, 20.0, 65.0, half_v,
                                   36.0 * M_PI / 180.0, 0.65, 12.0, fov,
                                   0.0, 3.0, 5.0, 12.0, 2.0, 0.15, 68.0, 30.0);
        c.set_gimbal(true, 0.0);
        c.set_vertical_climb(20.0, 20.0, 1.5, 2.0);
        c.set_guidance_tuning(2.0, 4.0, 12.0, 0.0, 0.5, 10.0, 0.0);
        return c;
    }

    std::vector<double> command(AdvancedHybridController &c, double dz, double offset,
                                double time = 0.0, double vz = 0.0,
                                const Eigen::Matrix3d &rotation = Eigen::Matrix3d::Identity())
    {
        const Eigen::Vector3d relative(30.0, 0.0, dz);
        const double elevation = std::atan2(dz, 30.0);
        return c.compute_control(rotation, 0.0, 0.0, relative, Eigen::Vector3d::Zero(),
                                 elevation, time, 0.0, 35.0, vz, 0.0, 0.0,
                                 Eigen::Vector3d::Zero(), elevation + offset);
    }
}

int main()
{
    constexpr double camera_offset = 20.0 * M_PI / 180.0;
    auto baseline = controller(false);
    const auto climb = command(baseline, 10.0, camera_offset);
    require(climb[0] > 9.81, "baseline must request ground climb");
    auto corrected = controller();
    for (int tick = 0; tick < 200; ++tick) {
        const auto result = command(corrected, 10.0, camera_offset, tick * 0.01);
        require(std::abs(result[0] - climb[0]) < 1e-10,
                "upper target below boresight must not suppress climb");
    }
    auto centered = controller();
    require(std::abs(command(centered, 10.0, 0.0)[0] - climb[0]) < 1e-10,
            "centered upper target behavior changed");
    for (double offset : {-camera_offset, camera_offset}) {
        corrected = controller();
        require(std::abs(command(corrected, 0.0, offset)[0] - 9.81) < 1e-10,
                "co-altitude camera pointing error must not inject climb or descent");
    }

    baseline = controller(false);
    const auto descent = command(baseline, -10.0, -camera_offset);
    corrected = controller();
    require(std::abs(command(corrected, -10.0, -camera_offset)[0] - descent[0]) < 1e-10,
            "lower target above boresight must not turn descent into climb");

    corrected = controller();
    require(command(corrected, -10.0, camera_offset)[0] == 2.0,
            "normal lower-target descent must remain available");
    corrected = controller();
    require(command(corrected, 10.0, -camera_offset)[0] >= 9.81,
            "normal upper-target FOV climb must remain available");

    baseline = controller(false);
    corrected = controller();
    const auto brake = command(baseline, 1.0, camera_offset, 0.0, 4.0);
    require(brake[0] < 9.81, "baseline must brake an existing climb");
    require(std::abs(command(corrected, 1.0, camera_offset, 0.0, 4.0)[0] - brake[0]) < 1e-10,
            "vertical braking must not be replaced with a takeoff thrust floor");

    for (double roll : {0.0, 0.5, 1.0}) {
        const Eigen::Matrix3d rotation = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
        baseline = controller(false);
        corrected = controller();
        require(std::abs(command(corrected, 10.0, camera_offset, 0.0, 0.0, rotation)[0] -
                         command(baseline, 10.0, camera_offset, 0.0, 0.0, rotation)[0]) < 1e-10,
                "actual-attitude thrust compensation changed");
    }

    // ---- Ground-launch boost ---------------------------------------------------------------
    auto boost = controller();
    boost.set_launch_boost(8.0, 3.0, 3.0, 25.0);
    boost.begin_ground_launch();
    require(std::abs(command(boost, 0.0, camera_offset)[0] - 17.81) < 1e-9,
            "launch boost must enforce the launch accel on a co-altitude target");
    require(command(boost, -10.0, camera_offset, 0.1)[0] > 17.0,
            "launch boost must supersede even the below-target descent path");
    // Exit on climb rate: a fresh controller at v_z=4 (>= exit speed) immediately follows
    // normal guidance (braking the 4 m/s climb -> the 2.0 floor), not the 17.81 boost.
    auto exit_c = controller();
    exit_c.set_launch_boost(8.0, 3.0, 3.0, 25.0);
    exit_c.begin_ground_launch();
    require(std::abs(command(exit_c, 0.0, 0.0, 0.0, 4.0)[0] - 2.0) < 1e-9,
            "launch boost must exit once the initial climb speed is built");
    boost.begin_ground_launch();
    require(command(boost, 0.0, 0.0, 0.0)[0] > 17.0, "launch boost re-arms");
    require(std::abs(command(boost, 0.0, 0.0, 4.0)[0] - 9.81) < 1e-9,
            "launch boost must expire at max_duration");
    boost.begin_ground_launch();
    boost.cancel_launch();
    require(std::abs(command(boost, 0.0, 0.0, 0.0)[0] - 9.81) < 1e-9, "cancel disarms boost");
    boost.begin_ground_launch();
    boost.reset_dynamic_state();
    require(std::abs(command(boost, 0.0, 0.0, 0.0)[0] - 9.81) < 1e-9, "reset disarms boost");
    auto plain = controller();
    plain.set_launch_boost(8.0, 3.0, 3.0, 25.0);
    require(std::abs(command(plain, 0.0, 0.0)[0] - 9.81) < 1e-9,
            "boost must not fire without a ground launch (airborne handover)");
    auto capped = controller();
    capped.set_guidance_tuning(10.0, 4.0, 12.0, 0.0, 0.5, 10.0, 0.0);
    capped.set_launch_boost(8.0, 3.0, 3.0, 25.0);
    capped.begin_ground_launch();
    const auto capped_cmd = command(capped, 10.0, 0.0);
    auto free_c = controller();
    free_c.set_guidance_tuning(10.0, 4.0, 12.0, 0.0, 0.5, 10.0, 0.0);
    const auto free_cmd = command(free_c, 10.0, 0.0);
    const double capped_rate = std::hypot(capped_cmd[1], std::hypot(capped_cmd[2], capped_cmd[3]));
    const double free_rate = std::hypot(free_cmd[1], std::hypot(free_cmd[2], free_cmd[3]));
    require(capped_cmd[0] > 17.0, "boost thrust holds under horizontal demand");
    require(capped_rate < free_rate, "launch tilt cap must reduce attitude rates");

    // ---- 12-06-28 round 2 regression: over-tilt demand through the launch cap --------------
    // The exact flight state: body 44.6 deg nose-down on the launcher, target reconstructed
    // below-forward at r=(18, 0, -27) m (elevation -56.3 deg) drives tilt_raw > max_tilt, so
    // the regular tilt cap replaces d with a dimensionless unit vector. In that regime the
    // launch boost must still enforce its angular cap with consistent units: desired tilt
    // re-clamped to launch_max_tilt (25 deg < the current 44.6 deg) and d rebuilt from the
    // boosted a_z, so the pitch-rate command turns nose-UP instead of pitching further
    // nose-down toward the 65 deg cap (the unfixed controller reproduces the logged +1.81
    // rad/s nose-down and 24.97 m/s^2 thrust on this exact input).
    const Eigen::Matrix3d nose_down =
        Eigen::AngleAxisd(44.6 * M_PI / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Vector3d below_forward(18.0, 0.0, -27.0);
    const double el_below = std::atan2(-27.0, 18.0);
    auto overtilt = [&](AdvancedHybridController &c) {
        return c.compute_control(nose_down, 0.0, 0.0, below_forward, Eigen::Vector3d::Zero(),
                                 el_below, 0.0, 0.0, 35.0, 0.0, 0.0, 0.0,
                                 Eigen::Vector3d::Zero(), el_below);
    };
    auto free_over = controller();
    free_over.set_guidance_tuning(10.0, 4.0, 12.0, 0.0, 0.5, 10.0, 0.0);
    const auto free_over_cmd = overtilt(free_over);
    require(free_over_cmd[2] > 0.5,
            "over-tilt scenario must command further nose-down without the boost");
    auto boost_over = controller();
    boost_over.set_guidance_tuning(10.0, 4.0, 12.0, 0.0, 0.5, 10.0, 0.0);
    boost_over.set_launch_boost(8.0, 3.0, 3.0, 25.0);
    boost_over.begin_ground_launch();
    const auto boost_over_cmd = overtilt(boost_over);
    require(boost_over_cmd[0] > 17.0, "over-tilt launch boost thrust holds");
    require(boost_over_cmd[2] < -0.5,
            "launch tilt cap must reverse the over-tilt pitch demand (12-06-28 round 2 regression)");
    // The same over-tilt state with a 35 deg cap (the deployed YAML value) must also bind:
    // 35 deg < 44.6 deg current tilt still commands nose-up, just gentler.
    auto boost35 = controller();
    boost35.set_guidance_tuning(10.0, 4.0, 12.0, 0.0, 0.5, 10.0, 0.0);
    boost35.set_launch_boost(8.0, 3.0, 3.0, 35.0);
    boost35.begin_ground_launch();
    const auto boost35_cmd = overtilt(boost35);
    require(boost35_cmd[2] < 0.0 && boost35_cmd[0] > 17.0,
            "35 deg launch cap must also bind in the over-tilt regime");
    bool invalid_launch_rejected = false;
    try {
        plain.set_launch_boost(-1.0, 3.0, 3.0, 25.0);
    } catch (const std::invalid_argument&) {
        invalid_launch_rejected = true;
    }
    require(invalid_launch_rejected, "invalid launch boost parameters accepted");
}
