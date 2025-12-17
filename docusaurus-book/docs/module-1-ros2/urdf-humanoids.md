---
sidebar_position: 4
title: 'URDF for Humanoid Robots'
---

# URDF for Humanoid Robots

Unified Robot Description Format (URDF) is an XML format for representing a robot model. This lesson covers how to create and use URDF files specifically for humanoid robots.

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML format used in ROS to describe robot models. It defines the physical and visual properties of a robot, including links, joints, and materials.

## Basic URDF Structure

A basic URDF file has the following structure:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links define the rigid bodies of the robot -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.25 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_link">
    <!-- ... -->
  </link>
</robot>
```

## URDF for Humanoid Robots

Humanoid robots have specific characteristics that require special attention in URDF:

### 1. Multi-Link Structure

Humanoid robots typically have:
- A torso/chest link
- A head link
- Two arms (each with shoulder, elbow, wrist joints)
- Two legs (each with hip, knee, ankle joints)
- Hands and feet

### 2. Example Humanoid URDF

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="torso_to_head" type="fixed">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.35" rpy="0 0 0"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="1 0.8 0.6 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="torso_to_left_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Additional joints and links would continue in a similar pattern -->
</robot>
```

## Xacro for Complex Humanoids

For complex humanoid robots, Xacro (XML Macros) is often used to simplify URDF:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="arm_length" value="0.3" />
  <xacro:property name="arm_radius" value="0.05" />

  <!-- Macro for creating an arm -->
  <xacro:macro name="arm" params="side parent *origin">
    <joint name="${parent}_to_${side}_shoulder" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_upper_arm"/>
      <xacro:insert_block name="origin"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
    </joint>

    <link name="${side}_upper_arm">
      <visual>
        <geometry>
          <cylinder length="${arm_length}" radius="${arm_radius}"/>
        </geometry>
        <material name="blue">
          <color rgba="0 0 1 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder length="${arm_length}" radius="${arm_radius}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Use the macro -->
  <xacro:arm side="left" parent="torso">
    <origin xyz="0.15 0 0.2" rpy="0 0 0"/>
  </xacro:arm>

  <xacro:arm side="right" parent="torso">
    <origin xyz="-0.15 0 0.2" rpy="0 0 0"/>
  </xacro:arm>
</robot>
```

## Gazebo Integration

To use your URDF in Gazebo, add Gazebo-specific tags:

```xml
<gazebo reference="left_upper_arm">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
</gazebo>
```

## Best Practices for Humanoid URDFs

1. **Follow the Right-Hand Rule**: Use the right-hand rule for joint rotations
2. **Proper Inertial Properties**: Define realistic mass and inertia for stable simulation
3. **Use Fixed Joints for Rigid Connections**: Use type="fixed" for non-moving connections
4. **Appropriate Joint Limits**: Set realistic joint limits based on human anatomy
5. **Collision Avoidance**: Consider potential self-collisions in the design
6. **Standard Names**: Use standard link names when possible for compatibility with controllers

## Visualization Tools

Use these ROS tools to visualize and debug your URDF:

```bash
# Check URDF syntax
check_urdf my_robot.urdf

# Visualize in RViz
ros2 run rviz2 rviz2

# Visualize joint transforms
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```

URDF is fundamental for humanoid robotics as it provides the geometric and kinematic description necessary for simulation, visualization, and control.