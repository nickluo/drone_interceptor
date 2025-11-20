
import numpy as np
from typing import Tuple, Dict
from collections import deque

class CameraModel:
    """模拟相机,根据3D位置生成2D检测框信息,并检查FOV"""
    def __init__(self, focal_length, target_size, fov_deg, pos_dev, size_dev):
        self.f = focal_length
        self.target_size = target_size
        self.max_angle_rad = np.deg2rad(fov_deg / 2.0)
        self.pos_dev = pos_dev
        self.size_dev = size_dev

    def project(self, drone_state, target_pos) -> Tuple[Dict[str, float], bool]:
        """将3D世界中的目标投影到相机平面。"""
        drone_pos = drone_state[0:3]
        psi = drone_state[8]
        # 提取云台角度 (假设在状态向量的第13, 14, 15位: index 12, 13, 14)
        gimbal_r, gimbal_p, gimbal_y = drone_state[12], drone_state[13], drone_state[14]

        los_W = target_pos - drone_pos
        
        # R_W_G (World to Gimbal Base/Stabilized Frame) - 仅包含偏航角
        cpsi, spsi = np.cos(psi), np.sin(psi)
        R_W_G = np.array([[cpsi, spsi, 0], [-spsi, cpsi, 0], [0, 0, 1]])
        
        # R_G_C (Gimbal Base to Camera) - 云台旋转
        c1, s1 = np.cos(gimbal_y), np.sin(gimbal_y)
        c2, s2 = np.cos(gimbal_r), np.sin(gimbal_r)
        c3, s3 = np.cos(gimbal_p), np.sin(gimbal_p)
        
        # 旋转矩阵: Gimbal -> Camera (与ACADOS模型保持一致)
        R_G_C = np.array([
            [c1*c3-s1*s2*s3, s1*c3+c1*s2*s3, -c2*s3],
            [-s1*c2,          c1*c2,           s2],
            [c1*s3+s1*s2*c3, s1*s3-c1*s2*c3,  c2*c3]
        ])

        los_C = R_G_C @ R_W_G @ los_W
        
        dist_cam_x = los_C[0]
        if dist_cam_x <= 0.1: 
            return {'u': 0, 'v': 0, 'size': 0, 'azimuth': 0, 'elevation': 0}, False

        angle_to_center = np.arctan2(np.sqrt(los_C[1]**2 + los_C[2]**2), dist_cam_x)
        if angle_to_center > self.max_angle_rad: 
            return {'u': 0, 'v': 0, 'size': 0, 'azimuth': 0, 'elevation': 0}, False

        pixel_size = self.f * self.target_size / dist_cam_x
        bbox = {
            'size': pixel_size,
            'u': self.f * los_C[1] / dist_cam_x,
            'v': self.f * los_C[2] / dist_cam_x,
            'azimuth': np.arctan2(-los_C[2], dist_cam_x),
            'elevation': np.arctan2(-los_C[1], dist_cam_x)
        }
        return bbox, True

    def get_noisy_measurement(self, bbox):
        """为检测框添加高斯噪声"""
        noisy_bbox = bbox.copy()
        noisy_bbox['u'] += np.random.normal(0, self.pos_dev)
        noisy_bbox['v'] += np.random.normal(0, self.pos_dev)
        noisy_bbox['size'] += np.random.normal(0, self.size_dev)
        # 确保尺寸为正
        noisy_bbox['size'] = max(1.0, noisy_bbox['size'])
        noisy_bbox['azimuth'] = np.arctan(-noisy_bbox['v'] / self.f)
        noisy_bbox['elevation'] = np.arctan(-noisy_bbox['u'] / self.f)
        return noisy_bbox

class PerceptionSim:
    def __init__(self, camera, delay, dt):
        self.camera = camera
        self.delay = delay
        self.dt = dt
        self.delay_steps = int(delay / dt)
        self.delay_buffer = deque(maxlen=self.delay_steps + 1)

    def update(self, x_current, target_pos_true, target_vel_true, history):
        bbox, is_visible = self.camera.project(x_current, target_pos_true)
        noisy_bbox = self.camera.get_noisy_measurement(bbox) if is_visible else bbox
        # print(f'azimuth: {np.rad2deg(noisy_bbox['azimuth']):.2f} deg, elevation: {np.rad2deg(noisy_bbox['elevation']):.2f} deg, visible: {is_visible}')

        self.delay_buffer.append((x_current.copy(), noisy_bbox, is_visible))
        
        if len(self.delay_buffer) < self.delay_steps + 1:
            target_pos_true += target_vel_true * self.dt
            # 只有当history不为None时才记录数据
            if history is not None:
                history['x'].append(x_current.copy())
                history['target_true'].append(target_pos_true.copy())
            return False, target_pos_true
        return True, target_pos_true
    
    def get_measure(self):
        return self.delay_buffer[0]
