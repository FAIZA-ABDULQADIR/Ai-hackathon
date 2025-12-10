using UnityEngine;

/// <summary>
/// Basic robot controller for Unity robotics simulation
/// Demonstrates movement, sensor simulation, and physics interaction
/// </summary>
public class RobotController : MonoBehaviour
{
    [Header("Movement Settings")]
    public float moveSpeed = 5.0f;
    public float rotateSpeed = 100.0f;

    [Header("Sensor Settings")]
    public float sensorRange = 10.0f;
    public LayerMask obstacleLayer;

    [Header("Physics Settings")]
    public float maxForce = 100f;
    public float maxTorque = 50f;

    private Rigidbody rb;
    private Vector3 targetPosition;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        targetPosition = transform.position;
    }

    void Update()
    {
        // Basic movement controls
        HandleMovement();

        // Sensor simulation
        SimulateSensors();
    }

    void HandleMovement()
    {
        // Get input for movement
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");

        // Calculate movement direction
        Vector3 movement = new Vector3(horizontal, 0, vertical);
        movement = transform.TransformDirection(movement);
        movement *= moveSpeed * Time.deltaTime;

        // Apply movement
        if (rb != null)
        {
            rb.MovePosition(transform.position + movement);
        }
        else
        {
            transform.Translate(movement);
        }
    }

    void SimulateSensors()
    {
        // Raycast for obstacle detection (simulating LiDAR)
        RaycastHit hit;
        if (Physics.Raycast(transform.position, transform.forward, out hit, sensorRange, obstacleLayer))
        {
            Debug.DrawRay(transform.position, transform.forward * hit.distance, Color.red);
            // In a real implementation, this would publish sensor data to ROS
            Debug.Log("Obstacle detected at: " + hit.distance + "m");
        }
        else
        {
            Debug.DrawRay(transform.position, transform.forward * sensorRange, Color.green);
        }
    }

    /// <summary>
    /// Simulate camera sensor data capture
    /// </summary>
    public Texture2D CaptureCameraView()
    {
        // This would normally capture from a camera component
        // For simulation, we'll return a placeholder texture
        Texture2D texture = new Texture2D(640, 480, TextureFormat.RGB24, false);
        Color[] pixels = new Color[640 * 480];

        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = Color.gray; // Placeholder color
        }

        texture.SetPixels(pixels);
        texture.Apply();

        return texture;
    }

    void OnValidate()
    {
        // Ensure values are reasonable
        moveSpeed = Mathf.Clamp(moveSpeed, 0.1f, 20f);
        rotateSpeed = Mathf.Clamp(rotateSpeed, 10f, 200f);
        sensorRange = Mathf.Clamp(sensorRange, 1f, 50f);
    }
}