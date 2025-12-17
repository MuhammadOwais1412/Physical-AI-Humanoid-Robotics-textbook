---
title: Biped Locomotion
sidebar_label: Biped Locomotion
description: Advanced bipedal locomotion systems for humanoid robots using Isaac Sim and ROS
sidebar_position: 3
---

# Biped Locomotion

## Overview

Biped locomotion is one of the most challenging aspects of humanoid robotics, requiring sophisticated control algorithms to achieve stable walking, running, and other dynamic movements. This chapter explores the principles and implementation of bipedal locomotion systems, focusing on approaches that leverage NVIDIA Isaac Sim for simulation and ROS for control. We'll examine the biomechanics of human walking, control strategies, and implementation techniques for creating stable bipedal robots.

## Learning Objectives

- Understand the biomechanics and dynamics of bipedal locomotion
- Implement control algorithms for stable walking patterns
- Configure and tune locomotion controllers in simulation
- Integrate locomotion systems with perception and navigation
- Evaluate locomotion performance and stability metrics

## Biomechanics of Bipedal Walking

Bipedal locomotion involves complex interactions between multiple body segments, requiring precise control of joint torques and forces to maintain balance while moving forward. The human walking gait cycle consists of two main phases:

1. **Stance Phase** (60% of gait cycle): The foot is in contact with the ground
2. **Swing Phase** (40% of gait cycle): The foot is off the ground and moving forward

### Key Concepts in Biped Locomotion

- **Zero Moment Point (ZMP)**: The point where the net moment of the ground reaction forces is zero
- **Center of Mass (CoM)**: The weighted average position of all mass in the body
- **Capture Point**: The point where the CoM needs to be placed to stop locomotion
- **Linear Inverted Pendulum (LIP)**: Simplified model for analyzing balance and walking

## Control Strategies for Biped Locomotion

### 1. Model-Based Control Approaches

Model-based control approaches use mathematical models of the robot's dynamics to compute appropriate control commands:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float64MultiArray
import numpy as np
from scipy import linalg
import math

class BipedLocomotionController(Node):
    def __init__(self):
        super().__init__('biped_locomotion_controller')

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )

        # Publishers
        self.joint_command_pub = self.create_publisher(
            Float64MultiArray, '/joint_commands', 10
        )
        self.com_pub = self.create_publisher(
            Vector3, '/center_of_mass', 10
        )
        self.zmp_pub = self.create_publisher(
            Vector3, '/zero_moment_point', 10
        )

        # Robot parameters
        self.robot_mass = 50.0  # kg
        self.gravity = 9.81    # m/s^2
        self.com_height = 0.8  # m (center of mass height)

        # Joint names and positions
        self.joint_names = [
            'left_hip_roll', 'left_hip_pitch', 'left_hip_yaw',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_roll', 'right_hip_pitch', 'right_hip_yaw',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll'
        ]
        self.joint_positions = {name: 0.0 for name in self.joint_names}
        self.joint_velocities = {name: 0.0 for name in self.joint_names}

        # Walking parameters
        self.step_length = 0.3  # meters
        self.step_height = 0.05  # meters
        self.step_duration = 1.0  # seconds
        self.step_phase = 0.0  # 0.0 to 1.0
        self.stride_frequency = 0.5  # Hz

        # Desired velocity
        self.desired_linear_vel = 0.0
        self.desired_angular_vel = 0.0

        # Control parameters
        self.zmp_control_gain = 50.0
        self.com_control_gain = 10.0
        self.balance_control_gain = 100.0

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100 Hz

        self.get_logger().info('Biped Locomotion Controller initialized')

    def joint_state_callback(self, msg):
        """Update joint positions and velocities"""
        for i, name in enumerate(msg.name):
            if name in self.joint_positions:
                self.joint_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.joint_velocities[name] = msg.velocity[i]

    def imu_callback(self, msg):
        """Process IMU data for balance control"""
        # Extract orientation and angular velocity
        self.orientation = msg.orientation
        self.angular_velocity = msg.angular_velocity

    def cmd_vel_callback(self, msg):
        """Update desired velocity commands"""
        self.desired_linear_vel = msg.linear.x
        self.desired_angular_vel = msg.angular.z

    def control_loop(self):
        """Main control loop for biped locomotion"""
        # Calculate current CoM and ZMP
        com_pos = self.calculate_center_of_mass()
        zmp_pos = self.calculate_zero_moment_point(com_pos)

        # Publish CoM and ZMP for visualization
        com_msg = Vector3()
        com_msg.x, com_msg.y, com_msg.z = com_pos
        self.com_pub.publish(com_msg)

        zmp_msg = Vector3()
        zmp_msg.x, zmp_msg.y, zmp_msg.z = zmp_pos
        self.zmp_pub.publish(zmp_msg)

        # Generate walking pattern based on desired velocity
        walking_pattern = self.generate_walking_pattern()

        # Calculate desired joint positions using inverse kinematics
        desired_joints = self.inverse_kinematics(walking_pattern, com_pos)

        # Apply balance control to maintain stability
        balance_corrections = self.balance_control(com_pos, zmp_pos)

        # Combine walking pattern with balance corrections
        final_joints = self.combine_commands(desired_joints, balance_corrections)

        # Publish joint commands
        joint_cmd_msg = Float64MultiArray()
        joint_cmd_msg.data = list(final_joints.values())
        self.joint_command_pub.publish(joint_cmd_msg)

    def calculate_center_of_mass(self):
        """Calculate the center of mass position"""
        # Simplified CoM calculation
        # In practice, this would use the full kinematic model
        com_x = 0.0
        com_y = 0.0
        com_z = self.com_height

        # Add some movement based on walking pattern
        if abs(self.desired_linear_vel) > 0.01:
            com_x += 0.02 * math.sin(self.step_phase * 2 * math.pi)

        return [com_x, com_y, com_z]

    def calculate_zero_moment_point(self, com_pos):
        """Calculate the Zero Moment Point"""
        # ZMP = CoM projected to ground plane with compensation
        # ZMP_x = CoM_x - (CoM_z - z_ground) / g * CoM_accel_x
        # For simplicity, we'll use a simplified version
        zmp_x = com_pos[0] - (com_pos[2] - 0.0) / self.gravity * 0.5  # Assume some acceleration
        zmp_y = com_pos[1] - (com_pos[2] - 0.0) / self.gravity * 0.2
        zmp_z = 0.0  # ZMP is on ground plane

        return [zmp_x, zmp_y, zmp_z]

    def generate_walking_pattern(self):
        """Generate desired walking pattern based on velocity commands"""
        # Update step phase based on desired velocity
        if abs(self.desired_linear_vel) > 0.01:
            self.step_phase += abs(self.desired_linear_vel) / self.step_length * 0.01
        else:
            self.step_phase = 0.0

        # Keep phase in [0, 1]
        self.step_phase = self.step_phase % 1.0

        # Generate foot positions based on walking pattern
        left_foot = self.calculate_foot_position('left', self.step_phase)
        right_foot = self.calculate_foot_position('right', self.step_phase)

        return {
            'left_foot': left_foot,
            'right_foot': right_foot,
            'com_trajectory': self.calculate_com_trajectory()
        }

    def calculate_foot_position(self, foot, phase):
        """Calculate desired foot position based on gait phase"""
        # Simplified foot trajectory
        # In practice, this would be more sophisticated
        if foot == 'left':
            # Left foot trajectory
            if phase < 0.5:  # Left foot is swing phase
                # Generate swing trajectory
                swing_phase = phase * 2  # Normalize to [0, 1]
                x_offset = self.step_length * swing_phase
                y_offset = 0.1 if swing_phase < 0.5 else 0.1 * (1 - swing_phase)
                z_offset = self.step_height * math.sin(swing_phase * math.pi)
            else:  # Left foot is stance phase
                x_offset = self.step_length
                y_offset = 0.1
                z_offset = 0.0
        else:  # right foot
            if phase >= 0.5:  # Right foot is swing phase
                swing_phase = (phase - 0.5) * 2  # Normalize to [0, 1]
                x_offset = self.step_length * swing_phase
                y_offset = -0.1 if swing_phase < 0.5 else -0.1 * (1 - swing_phase)
                z_offset = self.step_height * math.sin(swing_phase * math.pi)
            else:  # Right foot is stance phase
                x_offset = 0.0
                y_offset = -0.1
                z_offset = 0.0

        return [x_offset, y_offset, z_offset]

    def calculate_com_trajectory(self):
        """Calculate desired CoM trajectory"""
        # Simplified CoM trajectory following the walking pattern
        com_x = self.step_length * self.step_phase
        com_y = 0.0  # Keep CoM centered laterally
        com_z = self.com_height  # Keep CoM at constant height

        return [com_x, com_y, com_z]

    def inverse_kinematics(self, walking_pattern, com_pos):
        """Calculate joint angles using inverse kinematics"""
        # Simplified inverse kinematics
        # In practice, this would use more sophisticated IK solvers
        joint_angles = {}

        # Calculate leg positions based on foot targets
        left_foot_pos = walking_pattern['left_foot']
        right_foot_pos = walking_pattern['right_foot']

        # Calculate joint angles for left leg (simplified)
        joint_angles['left_hip_pitch'] = self.calculate_leg_ik(
            left_foot_pos, 'left'
        )
        joint_angles['left_knee'] = self.calculate_knee_angle(
            left_foot_pos, 'left'
        )
        joint_angles['left_ankle_pitch'] = self.calculate_ankle_angle(
            left_foot_pos, 'left'
        )

        # Calculate joint angles for right leg (simplified)
        joint_angles['right_hip_pitch'] = self.calculate_leg_ik(
            right_foot_pos, 'right'
        )
        joint_angles['right_knee'] = self.calculate_knee_angle(
            right_foot_pos, 'right'
        )
        joint_angles['right_ankle_pitch'] = self.calculate_ankle_angle(
            right_foot_pos, 'right'
        )

        # Set default angles for other joints
        joint_angles['left_hip_roll'] = 0.0
        joint_angles['left_hip_yaw'] = 0.0
        joint_angles['left_ankle_roll'] = 0.0
        joint_angles['right_hip_roll'] = 0.0
        joint_angles['right_hip_yaw'] = 0.0
        joint_angles['right_ankle_roll'] = 0.0

        return joint_angles

    def calculate_leg_ik(self, foot_pos, leg_side):
        """Calculate simplified inverse kinematics for leg"""
        # Simplified calculation - in practice, use proper IK solver
        # This is just a placeholder
        if leg_side == 'left':
            return 0.1 * foot_pos[0]  # Simplified relationship
        else:
            return 0.1 * foot_pos[0]

    def calculate_knee_angle(self, foot_pos, leg_side):
        """Calculate knee angle for desired foot position"""
        # Simplified calculation
        if leg_side == 'left':
            return 0.2 * foot_pos[2]  # Knee flexion based on height
        else:
            return 0.2 * foot_pos[2]

    def calculate_ankle_angle(self, foot_pos, leg_side):
        """Calculate ankle angle for desired foot position"""
        # Simplified calculation
        if leg_side == 'left':
            return -0.1 * foot_pos[1]  # Ankle adjustment based on lateral position
        else:
            return -0.1 * foot_pos[1]

    def balance_control(self, com_pos, zmp_pos):
        """Apply balance control to maintain stability"""
        # Calculate error between desired and actual ZMP
        desired_zmp = [0.0, 0.0, 0.0]  # Ideally, ZMP should be under feet
        zmp_error = [
            desired_zmp[0] - zmp_pos[0],
            desired_zmp[1] - zmp_pos[1],
            0.0
        ]

        # Apply feedback control to correct ZMP error
        balance_corrections = {
            'left_ankle_roll': self.zmp_control_gain * zmp_error[1],
            'right_ankle_roll': -self.zmp_control_gain * zmp_error[1],
            'left_ankle_pitch': self.zmp_control_gain * zmp_error[0],
            'right_ankle_pitch': self.zmp_control_gain * zmp_error[0],
        }

        return balance_corrections

    def combine_commands(self, desired_joints, balance_corrections):
        """Combine walking pattern commands with balance corrections"""
        final_joints = desired_joints.copy()

        # Apply balance corrections to appropriate joints
        for joint, correction in balance_corrections.items():
            if joint in final_joints:
                final_joints[joint] += correction

        return final_joints
```

### 2. Capture Point Control

The Capture Point is a critical concept in bipedal locomotion that indicates where the CoM needs to be placed to stop locomotion:

```python
import numpy as np
from scipy.integrate import odeint
import math

class CapturePointController:
    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = math.sqrt(gravity / com_height)

    def calculate_capture_point(self, com_pos, com_vel):
        """
        Calculate the capture point given CoM position and velocity
        Capture Point = CoM + CoM_velocity / omega
        """
        cp_x = com_pos[0] + com_vel[0] / self.omega
        cp_y = com_pos[1] + com_vel[1] / self.omega
        return [cp_x, cp_y]

    def is_stable(self, com_pos, com_vel, support_polygon):
        """
        Check if the robot is in a stable state
        Support polygon is typically the convex hull of contact points
        """
        capture_point = self.calculate_capture_point(com_pos, com_vel)

        # Check if capture point is within support polygon
        # This is a simplified check - in practice, use proper polygon inclusion
        cp_x, cp_y = capture_point
        poly_x, poly_y = support_polygon

        # Simplified bounding box check
        min_x, max_x = min(poly_x), max(poly_x)
        min_y, max_y = min(poly_y), max(poly_y)

        return min_x <= cp_x <= max_x and min_y <= cp_y <= max_y

    def generate_safe_footstep(self, current_com_pos, current_com_vel, step_width=0.2):
        """
        Generate a safe footstep location based on capture point
        """
        current_cp = self.calculate_capture_point(current_com_pos, current_com_vel)

        # Place next footstep near the capture point but within safe limits
        # This ensures the robot can stop if needed
        foot_x = current_cp[0]  # Place foot near capture point
        foot_y = -step_width if current_com_pos[1] > 0 else step_width  # Alternate sides

        return [foot_x, foot_y, 0.0]  # z=0 for ground contact
```

### 3. Linear Inverted Pendulum Model (LIP)

The Linear Inverted Pendulum model is a simplified representation used for balance control:

```python
class LinearInvertedPendulumController:
    def __init__(self, com_height=0.8, gravity=9.81, dt=0.01):
        self.com_height = com_height
        self.gravity = gravity
        self.dt = dt
        self.omega = math.sqrt(gravity / com_height)

        # State: [x, y, x_dot, y_dot] (CoM position and velocity)
        self.state = np.array([0.0, 0.0, 0.0, 0.0])

        # Reference trajectory
        self.reference_trajectory = []

    def update_state(self, com_pos, com_vel):
        """Update the current state of the inverted pendulum"""
        self.state = np.array([
            com_pos[0], com_pos[1],  # x, y position
            com_vel[0], com_vel[1]   # x, y velocity
        ])

    def calculate_zmp(self, state):
        """
        Calculate ZMP from current state
        ZMP = CoM - (CoM_height / gravity) * CoM_acceleration
        For LIP: CoM_acceleration = omega^2 * (CoM - ZMP)
        Therefore: ZMP = CoM - (1/omega^2) * CoM_acceleration
        """
        x, y, x_dot, y_dot = state
        zmp_x = x - x_dot / self.omega
        zmp_y = y - y_dot / self.omega
        return [zmp_x, zmp_y]

    def compute_control(self, desired_com_pos, desired_com_vel):
        """
        Compute control inputs to track desired CoM trajectory
        """
        current_x, current_y, current_x_dot, current_y_dot = self.state

        # Simple PD control on CoM position and velocity
        kp_pos = 10.0
        kd_vel = 2.0 * math.sqrt(10.0)  # Critical damping

        # Calculate control based on LIP dynamics
        control_x = kp_pos * (desired_com_pos[0] - current_x) + \
                   kd_vel * (desired_com_vel[0] - current_x_dot)
        control_y = kp_pos * (desired_com_pos[1] - current_y) + \
                   kd_vel * (desired_com_vel[1] - current_y_dot)

        # Convert control to joint torques through appropriate mapping
        return [control_x, control_y]

    def predict_motion(self, steps=100):
        """
        Predict future CoM motion using LIP model
        """
        predictions = []
        current_state = self.state.copy()

        for i in range(steps):
            # LIP dynamics: x_ddot = omega^2 * (x - zmp)
            # For balance, we want to control ZMP placement
            zmp_x, zmp_y = self.calculate_zmp(current_state)

            # Simple balance control: move ZMP toward desired location
            desired_zmp_x = 0.0  # Center of support polygon
            desired_zmp_y = 0.0

            # Apply control to move ZMP toward desired location
            control_x = 5.0 * (desired_zmp_x - zmp_x)
            control_y = 5.0 * (desired_zmp_y - zmp_y)

            # Update state using LIP dynamics
            x_ddot = self.omega**2 * (current_state[0] - zmp_x) + control_x
            y_ddot = self.omega**2 * (current_state[1] - zmp_y) + control_y

            # Integrate dynamics
            current_state[0] += current_state[2] * self.dt
            current_state[1] += current_state[3] * self.dt
            current_state[2] += x_ddot * self.dt
            current_state[3] += y_ddot * self.dt

            predictions.append(current_state.copy())

        return predictions
```

## Advanced Control Techniques

### Model Predictive Control (MPC) for Walking

Model Predictive Control can be used for more sophisticated walking control:

```python
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

class MPCWalkingController:
    def __init__(self, horizon=20, dt=0.1, com_height=0.8):
        self.horizon = horizon
        self.dt = dt
        self.com_height = com_height
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / com_height)

        # Walking parameters
        self.step_length = 0.3
        self.step_width = 0.2
        self.nominal_com_height = com_height

    def setup_optimization_problem(self, current_state, reference_trajectory):
        """
        Set up the MPC optimization problem
        State: [x, y, z, x_dot, y_dot, z_dot]
        """
        # Decision variables: footstep positions and CoM trajectory
        X = cp.Variable((6, self.horizon))  # State trajectory [x, y, z, x_dot, y_dot, z_dot]
        U = cp.Variable((2, self.horizon))  # Control inputs [zmp_x, zmp_y]

        # Cost function components
        cost = 0

        # Tracking cost - minimize deviation from reference trajectory
        for k in range(self.horizon):
            if k < len(reference_trajectory):
                ref_state = reference_trajectory[k]
                cost += cp.sum_squares(X[:3, k] - ref_state[:3]) * 1.0  # Position tracking
                cost += cp.sum_squares(X[3:, k] - ref_state[3:]) * 0.1  # Velocity tracking

        # Control effort cost - minimize ZMP changes
        for k in range(self.horizon - 1):
            cost += cp.sum_squares(U[:, k+1] - U[:, k]) * 0.01

        # Stability cost - keep ZMP within support polygon
        for k in range(self.horizon):
            # Simplified: keep ZMP near center
            cost += cp.sum_squares(U[:, k]) * 0.5

        # Dynamics constraints (Linear Inverted Pendulum Model)
        constraints = []

        # Initial state constraint
        constraints.append(X[:, 0] == current_state)

        # LIP dynamics: [x_ddot, y_ddot, z_ddot] = omega^2 * ([x, y, z] - [zmp_x, zmp_y, z_ground])
        for k in range(self.horizon - 1):
            # Simplified: assume constant CoM height (z_ddot = 0)
            constraints.append(X[2, k] == self.nominal_com_height)  # Constant height
            constraints.append(X[5, k] == 0.0)  # Zero vertical velocity

            # Horizontal dynamics
            constraints.append(X[0, k+1] == X[0, k] + X[3, k] * self.dt)
            constraints.append(X[1, k+1] == X[1, k] + X[4, k] * self.dt)

            # Acceleration based on ZMP
            constraints.append(X[3, k+1] == X[3, k] +
                              (self.omega**2 * (X[0, k] - U[0, k])) * self.dt)
            constraints.append(X[4, k+1] == X[4, k] +
                              (self.omega**2 * (X[1, k] - U[1, k])) * self.dt)

        # Support polygon constraints (simplified as bounds)
        for k in range(self.horizon):
            constraints.append(cp.abs(U[0, k]) <= 0.1)  # X bounds
            constraints.append(cp.abs(U[1, k]) <= 0.1)  # Y bounds

        # Formulate and solve the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        return problem, X, U

    def compute_control(self, current_state, reference_trajectory):
        """
        Compute the optimal control sequence using MPC
        """
        problem, X, U = self.setup_optimization_problem(current_state, reference_trajectory)

        try:
            problem.solve(solver=cp.ECOS, verbose=False)

            if problem.status == cp.OPTIMAL:
                # Return the first control action
                optimal_zmp = U[:, 0].value
                return optimal_zmp
            else:
                # Fallback to simple control if optimization fails
                return np.array([0.0, 0.0])
        except Exception as e:
            self.get_logger().error(f'MPC optimization failed: {e}')
            return np.array([0.0, 0.0])

    def get_logger(self):
        """Mock logger for this class"""
        class MockLogger:
            def info(self, msg):
                print(f'INFO: {msg}')
            def error(self, msg):
                print(f'ERROR: {msg}')
        return MockLogger()
```

## Integration with Isaac Sim

To integrate locomotion control with Isaac Sim, we need to properly interface with the simulation environment:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped

class IsaacSimLocomotionInterface(Node):
    def __init__(self):
        super().__init__('isaac_sim_locomotion_interface')

        # Initialize TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publishers for Isaac Sim joint control
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory', 10
        )

        # Publishers for visualization
        self.com_visualization_pub = self.create_publisher(
            Vector3, '/center_of_mass_visualization', 10
        )
        self.support_polygon_pub = self.create_publisher(
            Vector3, '/support_polygon_visualization', 10
        )

        # Subscribers for robot state
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # Subscribers for high-level commands
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )

        # Internal state
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self.imu_data = None
        self.desired_velocity = Twist()

        # Locomotion controller
        self.locomotion_controller = BipedLocomotionController()

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info('Isaac Sim Locomotion Interface initialized')

    def joint_state_callback(self, msg):
        """Process joint state messages from Isaac Sim"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]

        # Update locomotion controller with current state
        self.locomotion_controller.joint_positions = self.current_joint_positions
        self.locomotion_controller.joint_velocities = self.current_joint_velocities

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = msg
        self.locomotion_controller.imu_callback(msg)

    def cmd_vel_callback(self, msg):
        """Process velocity commands"""
        self.desired_velocity = msg
        self.locomotion_controller.cmd_vel_callback(msg)

    def control_loop(self):
        """Main control loop that interfaces with Isaac Sim"""
        # Update locomotion controller
        self.locomotion_controller.control_loop()

        # Generate joint trajectory command for Isaac Sim
        joint_trajectory = self.generate_joint_trajectory()

        # Publish joint trajectory to Isaac Sim
        self.joint_trajectory_pub.publish(joint_trajectory)

        # Publish visualization data
        self.publish_visualization_data()

        # Publish transforms for TF tree
        self.publish_transforms()

    def generate_joint_trajectory(self):
        """Generate joint trajectory message for Isaac Sim"""
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.header.frame_id = "base_link"

        # Set joint names
        trajectory.joint_names = list(self.locomotion_controller.joint_positions.keys())

        # Create trajectory point
        point = JointTrajectoryPoint()

        # Set positions (from locomotion controller)
        positions = []
        for joint_name in trajectory.joint_names:
            if joint_name in self.locomotion_controller.joint_positions:
                positions.append(self.locomotion_controller.joint_positions[joint_name])
            else:
                positions.append(0.0)  # Default position

        point.positions = positions

        # Set velocities
        velocities = []
        for joint_name in trajectory.joint_names:
            if joint_name in self.locomotion_controller.joint_velocities:
                velocities.append(self.locomotion_controller.joint_velocities[joint_name])
            else:
                velocities.append(0.0)  # Default velocity

        point.velocities = velocities

        # Set accelerations
        accelerations = [0.0] * len(positions)  # Simplified
        point.accelerations = accelerations

        # Set time from start
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 10000000  # 10ms

        trajectory.points = [point]
        return trajectory

    def publish_visualization_data(self):
        """Publish visualization data for RViz"""
        # Publish CoM position
        com_msg = Vector3()
        com_pos = [0.0, 0.0, 0.8]  # Simplified CoM position
        com_msg.x, com_msg.y, com_msg.z = com_pos
        self.com_visualization_pub.publish(com_msg)

    def publish_transforms(self):
        """Publish transforms for the robot"""
        # Publish base link transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"

        # Simplified: just publish a basic transform
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.8
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)
```

## Walking Pattern Generation

Creating stable walking patterns is crucial for biped locomotion:

```python
import numpy as np
import math

class WalkingPatternGenerator:
    def __init__(self, step_length=0.3, step_height=0.05, step_duration=1.0,
                 stance_duration_ratio=0.6, step_width=0.2):
        self.step_length = step_length
        self.step_height = step_height
        self.step_duration = step_duration
        self.stance_duration_ratio = stance_duration_ratio
        self.step_width = step_width

        # Gait phase tracking
        self.current_phase = 0.0
        self.current_support_foot = 'left'  # 'left' or 'right'
        self.step_count = 0

    def update_phase(self, dt, forward_speed):
        """Update gait phase based on forward speed"""
        if abs(forward_speed) > 0.01:
            phase_increment = (abs(forward_speed) / self.step_length) * dt
            self.current_phase += phase_increment

            # Switch support foot when phase crosses 0.5
            if self.current_phase >= 0.5 and self.current_support_foot == 'left':
                self.current_support_foot = 'right'
                self.step_count += 1
            elif self.current_phase >= 1.0:
                self.current_support_foot = 'left'
                self.current_phase = self.current_phase - 1.0
                self.step_count += 1
        else:
            self.current_phase = 0.0

    def calculate_foot_trajectory(self, support_foot, phase, forward_speed, turn_rate=0.0):
        """Calculate 3D trajectory for a foot during walking"""
        # Normalize phase within the current step cycle
        if support_foot == self.current_support_foot:
            # Support foot - remains on ground or follows ground profile
            x = 0.0  # Support foot stays approximately at origin
            y = self.step_width / 2 if support_foot == 'left' else -self.step_width / 2
            z = 0.0  # On ground
        else:
            # Swing foot - follows trajectory from behind support to ahead
            # Calculate swing phase (0 to 1)
            if self.current_support_foot == 'left':
                if support_foot == 'right':
                    swing_phase = phase * 2 if phase < 0.5 else (phase - 0.5) * 2
            else:  # current_support_foot == 'right'
                if support_foot == 'left':
                    swing_phase = phase * 2 if phase < 0.5 else (phase - 0.5) * 2

            # Ensure swing phase is in [0, 1]
            if phase < 0.5 and support_foot != self.current_support_foot:
                swing_phase = phase * 2
            elif phase >= 0.5 and support_foot != self.current_support_foot:
                swing_phase = (phase - 0.5) * 2
            else:
                swing_phase = 0.0

            # Calculate foot trajectory
            x = self.step_length * (swing_phase - 0.5)  # Move from -step_length/2 to +step_length/2
            y = self.step_width / 2 if support_foot == 'left' else -self.step_width / 2
            z = self.step_height * math.sin(swing_phase * math.pi)  # Vertical lift

            # Add forward progression
            x += forward_speed * self.step_duration * (self.current_phase if self.current_support_foot != support_foot else 0)

            # Add turning component
            if turn_rate != 0.0:
                y_offset = turn_rate * self.step_duration * (self.current_phase if self.current_support_foot != support_foot else 0)
                y += y_offset

        return [x, y, z]

    def calculate_com_trajectory(self, phase, forward_speed, turn_rate=0.0):
        """Calculate CoM trajectory for stable walking"""
        # Generate CoM trajectory that maintains stability
        # Use 3rd order polynomial for smooth transitions

        # Forward position (follows walking progression)
        com_x = forward_speed * self.step_duration * phase

        # Lateral position (oscillates between feet for stability)
        lateral_oscillation = 0.02 * math.sin(phase * 2 * math.pi)  # Small lateral sway
        com_y = lateral_oscillation

        # Vertical position (maintains constant height with slight oscillation)
        com_z = 0.8 + 0.01 * math.sin(phase * 4 * math.pi)  # Small vertical oscillation

        # Add turning effect
        if turn_rate != 0.0:
            com_y += turn_rate * 0.1 * phase  # Slight lateral offset during turns

        return [com_x, com_y, com_z]

    def generate_step_sequence(self, num_steps, forward_speed, turn_rate=0.0):
        """Generate a sequence of steps"""
        step_sequence = []

        for step_idx in range(num_steps):
            step_info = {
                'step_number': step_idx,
                'left_foot_trajectory': [],
                'right_foot_trajectory': [],
                'com_trajectory': []
            }

            # Generate trajectories for this step
            for t in np.linspace(0, 1, 20):  # 20 points per step
                # Calculate foot positions
                left_foot = self.calculate_foot_trajectory('left', t, forward_speed, turn_rate)
                right_foot = self.calculate_foot_trajectory('right', t, forward_speed, turn_rate)
                com_pos = self.calculate_com_trajectory(t, forward_speed, turn_rate)

                step_info['left_foot_trajectory'].append(left_foot)
                step_info['right_foot_trajectory'].append(right_foot)
                step_info['com_trajectory'].append(com_pos)

            step_sequence.append(step_info)

        return step_sequence
```

## Performance Evaluation and Tuning

### Stability Metrics

```python
import numpy as np

class LocomotionEvaluator:
    def __init__(self):
        self.metrics_history = {
            'zmp_error': [],
            'com_deviation': [],
            'joint_effort': [],
            'step_regularity': []
        }

    def evaluate_stability(self, com_pos, zmp_pos, support_polygon):
        """Evaluate the stability of the current locomotion state"""
        # Calculate ZMP error (distance from desired ZMP)
        desired_zmp_x = np.mean(support_polygon[0]) if support_polygon else 0.0
        desired_zmp_y = np.mean(support_polygon[1]) if support_polygon else 0.0

        zmp_error = np.sqrt((zmp_pos[0] - desired_zmp_x)**2 + (zmp_pos[1] - desired_zmp_y)**2)

        # Calculate CoM deviation from nominal position
        nominal_com_y = 0.0  # Center between feet
        com_deviation = abs(com_pos[1] - nominal_com_y)

        # Store metrics
        self.metrics_history['zmp_error'].append(zmp_error)
        self.metrics_history['com_deviation'].append(com_deviation)

        return {
            'zmp_error': zmp_error,
            'com_deviation': com_deviation,
            'is_stable': zmp_error < 0.05 and com_deviation < 0.05  # Thresholds
        }

    def evaluate_step_quality(self, step_data):
        """Evaluate the quality of a completed step"""
        # Calculate step length consistency
        step_lengths = [step['length'] for step in step_data if 'length' in step]
        if len(step_lengths) > 1:
            length_variability = np.std(step_lengths) / np.mean(step_lengths)
        else:
            length_variability = 0.0

        # Calculate step width consistency
        step_widths = [step['width'] for step in step_data if 'width' in step]
        if len(step_widths) > 1:
            width_variability = np.std(step_widths) / np.mean(step_widths)
        else:
            width_variability = 0.0

        # Calculate step timing regularity
        step_times = [step['duration'] for step in step_data if 'duration' in step]
        if len(step_times) > 1:
            timing_variability = np.std(step_times) / np.mean(step_times)
        else:
            timing_variability = 0.0

        # Overall step quality score
        quality_score = 1.0 - (length_variability + width_variability + timing_variability) / 3.0

        return {
            'length_variability': length_variability,
            'width_variability': width_variability,
            'timing_variability': timing_variability,
            'quality_score': max(0.0, quality_score)
        }

    def get_performance_summary(self):
        """Get a summary of locomotion performance"""
        if not self.metrics_history['zmp_error']:
            return {'error': 'No data available'}

        avg_zmp_error = np.mean(self.metrics_history['zmp_error'])
        avg_com_deviation = np.mean(self.metrics_history['com_deviation'])
        max_zmp_error = np.max(self.metrics_history['zmp_error'])
        stability_percentage = sum(1 for err in self.metrics_history['zmp_error'] if err < 0.05) / len(self.metrics_history['zmp_error'])

        return {
            'average_zmp_error': avg_zmp_error,
            'average_com_deviation': avg_com_deviation,
            'max_zmp_error': max_zmp_error,
            'stability_percentage': stability_percentage,
            'total_samples': len(self.metrics_history['zmp_error'])
        }
```

## Self-Check Questions


## Summary

Biped locomotion represents one of the most challenging problems in robotics, requiring sophisticated control algorithms to achieve stable and efficient walking. Key aspects of successful biped locomotion include:

1. **Dynamic Balance**: Maintaining the Center of Mass within the support polygon using concepts like ZMP and Capture Point
2. **Control Strategies**: Implementing appropriate control algorithms such as Model Predictive Control, Capture Point control, or LIP-based controllers
3. **Walking Pattern Generation**: Creating stable gait patterns that alternate support between feet while maintaining balance
4. **Integration**: Properly interfacing with simulation environments like Isaac Sim and real hardware
5. **Performance Evaluation**: Continuously monitoring stability metrics and adjusting control parameters

Successful implementation of biped locomotion requires careful tuning of control parameters, understanding of robot dynamics, and integration with perception systems for adaptive walking on various terrains. The combination of model-based control approaches with simulation testing in Isaac Sim provides a robust framework for developing stable bipedal robots.