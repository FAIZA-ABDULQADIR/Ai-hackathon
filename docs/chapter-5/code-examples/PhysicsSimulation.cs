using UnityEngine;

/// <summary>
/// Physics simulation example for robotic manipulation
/// Demonstrates joint control and physics-based interactions
/// </summary>
public class PhysicsSimulation : MonoBehaviour
{
    [Header("Joint Configuration")]
    public ConfigurableJoint joint;
    public float targetPosition = 0f;
    public float stiffness = 100f;
    public float damping = 10f;

    [Header("Manipulation Settings")]
    public float maxForce = 100f;
    public float maxVelocity = 50f;

    [Header("Sensor Simulation")]
    public Transform sensorTransform;
    public float sensorRange = 5f;

    private JointDrive drive;
    private Rigidbody connectedBody;

    void Start()
    {
        if (joint == null)
        {
            joint = GetComponent<ConfigurableJoint>();
        }

        SetupJointDrive();
        connectedBody = joint.connectedBody;
    }

    void FixedUpdate()
    {
        ControlJoint();
        SimulateSensors();
    }

    void SetupJointDrive()
    {
        drive = new JointDrive();
        drive.positionSpring = stiffness;
        drive.positionDamper = damping;
        drive.maximumForce = maxForce;

        joint.slerpDrive = drive;
        joint.rotationDriveMode = RotationDriveMode.XYAndZ;
    }

    void ControlJoint()
    {
        // Set target rotation for the joint
        joint.targetRotation = Quaternion.Euler(targetPosition, 0, 0);
        joint.targetAngularVelocity = Vector3.zero;
    }

    void SimulateSensors()
    {
        // Simulate force/torque sensors
        if (joint != null)
        {
            Vector3 reactionForce = joint.reactionForce;
            Vector3 reactionTorque = joint.reactionTorque;

            // In a real implementation, this would publish force/torque data to ROS
            Debug.Log($"Force: {reactionForce}, Torque: {reactionTorque}");
        }

        // Raycast for proximity sensing
        if (sensorTransform != null)
        {
            RaycastHit hit;
            if (Physics.Raycast(sensorTransform.position, sensorTransform.forward, out hit, sensorRange))
            {
                Debug.DrawRay(sensorTransform.position, sensorTransform.forward * hit.distance, Color.blue);
                // This could represent distance sensor data
            }
            else
            {
                Debug.DrawRay(sensorTransform.position, sensorTransform.forward * sensorRange, Color.yellow);
            }
        }
    }

    /// <summary>
    /// Apply force to the connected body for manipulation
    /// </summary>
    /// <param name="force">Force vector to apply</param>
    public void ApplyManipulationForce(Vector3 force)
    {
        if (connectedBody != null)
        {
            connectedBody.AddForceAtPosition(force, transform.position);
        }
    }

    /// <summary>
    /// Get current joint state for digital twin synchronization
    /// </summary>
    /// <returns>Joint state information</returns>
    public JointState GetJointState()
    {
        return new JointState
        {
            position = joint.connectedAnchor,
            rotation = joint.connectedAnchor,
            velocity = joint.connectedBody != null ? joint.connectedBody.velocity : Vector3.zero,
            force = joint.reactionForce
        };
    }

    [System.Serializable]
    public struct JointState
    {
        public Vector3 position;
        public Vector3 rotation;
        public Vector3 velocity;
        public Vector3 force;
    }

    void OnValidate()
    {
        stiffness = Mathf.Clamp(stiffness, 0f, 1000f);
        damping = Mathf.Clamp(damping, 0f, 100f);
        maxForce = Mathf.Clamp(maxForce, 0f, 1000f);
        maxVelocity = Mathf.Clamp(maxVelocity, 0f, 100f);
        sensorRange = Mathf.Clamp(sensorRange, 0.1f, 20f);
    }
}