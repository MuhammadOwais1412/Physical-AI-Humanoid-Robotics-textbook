# Solutions for Module 2: The Digital Twin (Gazebo & Unity)

## Exercise 1: Gazebo Simulation
**Problem**: Create a simple Gazebo world with a robot model.

**Solution**:

### World File (simple_world.world)
```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="simple_world">
    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a light source -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add a simple box obstacle -->
    <model name="box_obstacle">
      <pose>2 2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.3 0.3 1</ambient>
            <diffuse>0.8 0.3 0.3 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Exercise 2: Unity Simulation Setup
**Problem**: Create a basic Unity scene with a robot model and basic navigation.

**Solution**:

```csharp
using UnityEngine;
using System.Collections;

public class RobotController : MonoBehaviour
{
    public float moveSpeed = 5.0f;
    public float rotateSpeed = 100.0f;

    void Update()
    {
        // Basic movement controls
        float translation = Input.GetAxis("Vertical") * moveSpeed * Time.deltaTime;
        float rotation = Input.GetAxis("Horizontal") * rotateSpeed * Time.deltaTime;

        transform.Translate(0, 0, translation);
        transform.Rotate(0, rotation, 0);
    }
}

// Navigation agent for autonomous movement
public class NavigationAgent : MonoBehaviour
{
    UnityEngine.AI.NavMeshAgent agent;
    public Transform target;

    void Start()
    {
        agent = GetComponent<UnityEngine.AI.NavMeshAgent>();
    }

    void Update()
    {
        if (target != null)
        {
            agent.SetDestination(target.position);
        }
    }
}
```

## Exercise 3: Sensor Simulation
**Problem**: Implement a simulated LiDAR sensor in Gazebo.

**Solution**:

### Robot with LiDAR Sensor (in URDF/SDF)
```xml
<!-- LiDAR sensor definition -->
<sensor name="lidar_sensor" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1.0</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>10.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

## Exercise 4: Human-Robot Interaction in Unity
**Problem**: Create a Unity interface for human-robot interaction.

**Solution**:

```csharp
using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class HRIInterface : MonoBehaviour
{
    public Text statusText;
    public Button moveButton;
    public Button stopButton;
    public Slider speedSlider;

    private RobotController robot;

    void Start()
    {
        moveButton.onClick.AddListener(MoveRobot);
        stopButton.onClick.AddListener(StopRobot);
        speedSlider.onValueChanged.AddListener(ChangeSpeed);
    }

    void MoveRobot()
    {
        // Send move command to robot
        statusText.text = "Robot moving...";
    }

    void StopRobot()
    {
        // Send stop command to robot
        statusText.text = "Robot stopped";
    }

    void ChangeSpeed(float speed)
    {
        // Update robot speed
        statusText.text = $"Speed: {speed:F2}";
    }
}
```