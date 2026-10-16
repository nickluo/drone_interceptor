"""ROS graph regressions; run with unittest in the sourced ROS workspace."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest

from ament_index_python.packages import get_package_prefix, get_package_share_directory
import rclpy
from rclpy.context import Context
import yaml


DEFAULT_TOPICS = {
    "imu": "/mavros/imu/data",
    "bbox": "/fpv/target_bbox",
    "gimbal": "/fpv/gimbal",
    "mission_launch": "/fpv/tracker_trigger",
    "mission": "/fpv/mission_launch",
    "control": "/fpv/control_command",
    "mission_state": "/fpv/mission_state",
}
SUBSCRIPTIONS = {
    "imu": "sensor_msgs/msg/Imu",
    "bbox": "vision_msgs/msg/Detection2D",
    "gimbal": "sensor_msgs/msg/Imu",
    "mission_launch": "std_msgs/msg/Header",
    "mission": "std_msgs/msg/Bool",
}
PUBLICATIONS = {
    "control": "quadrotor_msgs/msg/ControlCommand",
    "mission_state": "std_msgs/msg/Bool",
}


class RelativeImmUkfTopicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain = int(os.environ.get("ROS_DOMAIN_ID", "178"))
        cls.context = Context()
        rclpy.init(context=cls.context, domain_id=cls.domain)
        cls.probe = rclpy.create_node("relative_imm_ukf_topics_probe", context=cls.context)
        cls.binary = Path(get_package_prefix("drone_interceptor")) / (
            "lib/drone_interceptor/interceptor_node_relative_imm_ukf")

    @classmethod
    def tearDownClass(cls):
        cls.probe.destroy_node()
        rclpy.shutdown(context=cls.context)

    @contextmanager
    def running_node(self, extra=(), parameters=None):
        with tempfile.TemporaryDirectory(prefix="interceptor_topics_") as directory:
            command = [str(self.binary), "--ros-args"]
            if parameters is not None:
                path = Path(directory) / "parameters.yaml"
                path.write_text(json.dumps({"/**": {"ros__parameters": parameters}}))
                command += ["--params-file", str(path)]
            command += list(extra)
            with tempfile.TemporaryFile(mode="w+") as output:
                process = subprocess.Popen(
                    command, stdout=output, stderr=subprocess.STDOUT,
                    env={**os.environ, "ROS_DOMAIN_ID": str(self.domain)})
                try:
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        output.seek(0)
                        text = output.read()
                        if process.poll() is not None:
                            self.fail(f"controller exited {process.returncode}:\n{text}")
                        if "Relative IMM-UKF node ready" in text:
                            break
                        time.sleep(.05)
                    else:
                        self.fail(f"controller did not become ready:\n{text}")
                    yield
                finally:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                            self.fail("controller did not stop after SIGINT")

    def assert_topics(self, expected, namespace="/"):
        deadline = time.monotonic() + 5
        subscriptions, publications = {}, {}
        while time.monotonic() < deadline:
            if ("interceptor_node_relative_imm_ukf", namespace) not in (
                    self.probe.get_node_names_and_namespaces()):
                time.sleep(.05)
                continue
            subscriptions = dict(self.probe.get_subscriber_names_and_types_by_node(
                "interceptor_node_relative_imm_ukf", namespace))
            publications = dict(self.probe.get_publisher_names_and_types_by_node(
                "interceptor_node_relative_imm_ukf", namespace))
            if all(subscriptions.get(expected[key]) == [value]
                   for key, value in SUBSCRIPTIONS.items()) and all(
                       publications.get(expected[key]) == [value]
                       for key, value in PUBLICATIONS.items()):
                self.assertEqual(set(subscriptions) - {"/parameter_events"},
                                 {expected[key] for key in SUBSCRIPTIONS})
                self.assertEqual(set(publications) - {"/parameter_events", "/rosout"},
                                 {expected[key] for key in PUBLICATIONS})
                return
            time.sleep(.05)
        self.fail(f"expected {expected}; subscriptions={subscriptions}; publications={publications}")

    def test_defaults_preserve_existing_endpoint_names(self):
        with self.running_node():
            self.assert_topics(DEFAULT_TOPICS)

    def test_packaged_yaml_controls_endpoint_names(self):
        config = Path(get_package_share_directory("drone_interceptor")) / (
            "config/relative_imm_ukf_params.yaml")
        parameters = yaml.safe_load(config.read_text())[
            "interceptor_node_relative_imm_ukf"]["ros__parameters"]
        with self.running_node(parameters=parameters):
            self.assert_topics(parameters["topics"])

    def test_custom_topic_parameters_bind_all_endpoints(self):
        topics = {key: f"/configured/{key}" for key in DEFAULT_TOPICS}
        with self.running_node(parameters={"topics": topics}):
            self.assert_topics(topics)

    def test_relative_names_and_gimbal_remapping_follow_ros_rules(self):
        topics = {key: f"configured/{key}" for key in DEFAULT_TOPICS}
        expected = {key: f"/topic_test/{value}" for key, value in topics.items()}
        expected["gimbal"] = "/remapped/gimbal"
        with self.running_node(
                extra=("-r", "__ns:=/topic_test", "-r", "configured/gimbal:=/remapped/gimbal"),
                parameters={"topics": topics}):
            self.assert_topics(expected, "/topic_test")


if __name__ == "__main__":
    unittest.main()
