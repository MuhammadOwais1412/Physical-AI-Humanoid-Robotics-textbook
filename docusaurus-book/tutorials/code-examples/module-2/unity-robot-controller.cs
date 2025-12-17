using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class UnityRobotController : MonoBehaviour
{
    ROSConnection ros;

    // Robot parameters
    public float wheelRadius = 0.1f;
    public float wheelSeparation = 0.5f;

    // Wheel objects for visualization
    public GameObject leftWheel;
    public GameObject rightWheel;

    // Current robot state
    float leftWheelVelocity = 0f;
    float rightWheelVelocity = 0f;

    void Start()
    {
        ros = ROSConnection.instance;

        // Subscribe to velocity commands
        ros.Subscribe<TwistMsg>("cmd_vel", VelocityCommandCallback);

        // Start coroutine for updating robot state
        StartCoroutine(UpdateRobotState());
    }

    void VelocityCommandCallback(TwistMsg velocityCommand)
    {
        // Convert linear and angular velocities to wheel velocities
        float linearVel = velocityCommand.linear.x;
        float angularVel = velocityCommand.angular.z;

        // Differential drive kinematics
        leftWheelVelocity = (linearVel - angularVel * wheelSeparation / 2.0f) / wheelRadius;
        rightWheelVelocity = (linearVel + angularVel * wheelSeparation / 2.0f) / wheelRadius;
    }

    IEnumerator UpdateRobotState()
    {
        while (true)
        {
            // Update wheel positions based on velocities
            if (leftWheel != null)
            {
                leftWheel.transform.Rotate(Vector3.right, leftWheelVelocity * Mathf.Rad2Deg * Time.deltaTime);
            }

            if (rightWheel != null)
            {
                rightWheel.transform.Rotate(Vector3.right, rightWheelVelocity * Mathf.Rad2Deg * Time.deltaTime);
            }

            // Update robot position (simplified)
            float avgWheelVel = (leftWheelVelocity + rightWheelVelocity) / 2.0f;
            float angularVel = (rightWheelVelocity - leftWheelVelocity) / wheelSeparation;

            transform.Translate(Vector3.forward * avgWheelVel * Time.deltaTime * wheelRadius);
            transform.Rotate(Vector3.up, angularVel * Mathf.Rad2Deg * Time.deltaTime);

            yield return new WaitForEndOfFrame();
        }
    }

    void OnDisable()
    {
        if (ros != null)
        {
            ros.Unsubscribe("cmd_vel");
        }
    }
}