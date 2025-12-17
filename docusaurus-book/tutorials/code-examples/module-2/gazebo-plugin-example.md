# Gazebo Plugin Example

This example demonstrates how to create a custom Gazebo plugin for a robot controller.

## CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.5)
project(gazebo_custom_plugin)

# Find packages
find_package(gazebo REQUIRED)
find_package(ament_cmake REQUIRED)

# Include directories
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

# Add library
add_library(gazebo_robot_controller SHARED robot_controller_plugin.cpp)
target_link_libraries(gazebo_robot_controller ${GAZEBO_LIBRARIES})

# Install
install(TARGETS gazebo_robot_controller
  LIBRARY DESTINATION lib
)
```

## robot_controller_plugin.cpp

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

namespace gazebo
{
  class RobotController : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Store the model pointer for convenience
      this->model = _parent;

      // Get pointers to the joints
      this->leftJoint = this->model->GetJoint("left_wheel_hinge");
      this->rightJoint = this->model->GetJoint("right_wheel_hinge");

      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&RobotController::OnUpdate, this));
    }

    // Called by the world update start event
    public: void OnUpdate()
    {
      // Apply a small linear velocity to the first joint
      this->leftJoint->SetVelocity(0, 0.5);
      this->rightJoint->SetVelocity(0, 0.5);
    }

    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the joints
    private: physics::JointPtr leftJoint;
    private: physics::JointPtr rightJoint;

    // Event connection
    private: event::ConnectionPtr updateConnection;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(RobotController)
}
```

## Plugin Usage in URDF/SDF

To use this plugin in your robot model:

```xml
<gazebo>
  <plugin name="robot_controller" filename="libgazebo_robot_controller.so">
  </plugin>
</gazebo>
```