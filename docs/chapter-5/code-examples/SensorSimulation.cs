using UnityEngine;
using System.Collections;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;

/// <summary>
/// Unity sensor simulation integrated with ROS
/// Demonstrates camera, LiDAR, and IMU sensor simulation
/// </summary>
public class SensorSimulation : MonoBehaviour
{
    [Header("ROS Settings")]
    public string rosTopicPrefix = "/unity_robot";

    [Header("Camera Settings")]
    public Camera sensorCamera;
    public int cameraWidth = 640;
    public int cameraHeight = 480;

    [Header("LiDAR Settings")]
    public int lidarRays = 360;
    public float lidarRange = 20f;
    public float lidarAngle = 360f;

    [Header("IMU Settings")]
    public float imuNoise = 0.01f;

    private ROSConnection ros;
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private MeshRenderer meshRenderer;

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();

        // Setup camera sensor
        SetupCameraSensor();

        // Start sensor publishing coroutines
        StartCoroutine(PublishCameraData());
        StartCoroutine(PublishLidarData());
        StartCoroutine(PublishImuData());
    }

    void SetupCameraSensor()
    {
        if (sensorCamera == null)
            sensorCamera = GetComponent<Camera>();

        renderTexture = new RenderTexture(cameraWidth, cameraHeight, 24);
        sensorCamera.targetTexture = renderTexture;
        texture2D = new Texture2D(cameraWidth, cameraHeight, TextureFormat.RGB24, false);

        // Add a mesh renderer to visualize the camera's view in the editor
        meshRenderer = GetComponent<MeshRenderer>();
        if (meshRenderer == null)
        {
            GameObject quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            quad.transform.SetParent(transform);
            quad.transform.localPosition = Vector3.forward * 0.1f;
            meshRenderer = quad.GetComponent<MeshRenderer>();
        }
    }

    IEnumerator PublishCameraData()
    {
        while (true)
        {
            // Capture camera image
            RenderTexture.active = renderTexture;
            texture2D.ReadPixels(new Rect(0, 0, cameraWidth, cameraHeight), 0, 0);
            texture2D.Apply();
            RenderTexture.active = null;

            // Create ROS image message
            ImageMsg imageMsg = new ImageMsg
            {
                header = new HeaderMsg
                {
                    stamp = new TimeMsg(ROSConnection.GetGlobalTimestamp()),
                    frame_id = rosTopicPrefix + "/camera_frame"
                },
                height = (uint)cameraHeight,
                width = (uint)cameraWidth,
                encoding = "rgb8",
                is_bigendian = 0,
                step = (uint)(cameraWidth * 3), // 3 bytes per pixel for RGB
                data = texture2D.GetRawTextureData<byte>().ToArray()
            };

            // Publish to ROS
            ros.Publish(rosTopicPrefix + "/camera/image_raw", imageMsg);

            yield return new WaitForSeconds(0.1f); // 10 Hz
        }
    }

    IEnumerator PublishLidarData()
    {
        while (true)
        {
            // Create LiDAR message
            LaserScanMsg lidarMsg = new LaserScanMsg
            {
                header = new HeaderMsg
                {
                    stamp = new TimeMsg(ROSConnection.GetGlobalTimestamp()),
                    frame_id = rosTopicPrefix + "/lidar_frame"
                },
                angle_min = -lidarAngle * Mathf.Deg2Rad / 2,
                angle_max = lidarAngle * Mathf.Deg2Rad / 2,
                angle_increment = (lidarAngle * Mathf.Deg2Rad) / lidarRays,
                time_increment = 0,
                scan_time = 0.1f,
                range_min = 0.1f,
                range_max = lidarRange,
                ranges = new float[lidarRays],
                intensities = new float[lidarRays]
            };

            // Simulate LiDAR rays
            for (int i = 0; i < lidarRays; i++)
            {
                float angle = transform.eulerAngles.y * Mathf.Deg2Rad +
                             (i * (lidarAngle / lidarRays) - (lidarAngle / 2)) * Mathf.Deg2Rad;

                Vector3 direction = new Vector3(Mathf.Sin(angle), 0, Mathf.Cos(angle));

                RaycastHit hit;
                if (Physics.Raycast(transform.position, direction, out hit, lidarRange))
                {
                    lidarMsg.ranges[i] = hit.distance;
                    lidarMsg.intensities[i] = 1.0f; // Full intensity for valid readings
                }
                else
                {
                    lidarMsg.ranges[i] = lidarRange; // Max range if no obstacle
                    lidarMsg.intensities[i] = 0.0f;
                }
            }

            // Publish to ROS
            ros.Publish(rosTopicPrefix + "/scan", lidarMsg);

            yield return new WaitForSeconds(0.05f); // 20 Hz
        }
    }

    IEnumerator PublishImuData()
    {
        while (true)
        {
            // Create IMU message with some noise
            ImuMsg imuMsg = new ImuMsg
            {
                header = new HeaderMsg
                {
                    stamp = new TimeMsg(ROSConnection.GetGlobalTimestamp()),
                    frame_id = rosTopicPrefix + "/imu_frame"
                },
                orientation = new GeometryMsgs.Quaternion
                {
                    x = transform.rotation.x + Random.Range(-imuNoise, imuNoise),
                    y = transform.rotation.y + Random.Range(-imuNoise, imuNoise),
                    z = transform.rotation.z + Random.Range(-imuNoise, imuNoise),
                    w = transform.rotation.w + Random.Range(-imuNoise, imuNoise)
                },
                angular_velocity = new GeometryMsgs.Vector3
                {
                    x = Random.Range(-imuNoise, imuNoise),
                    y = Random.Range(-imuNoise, imuNoise),
                    z = Random.Range(-imuNoise, imuNoise)
                },
                linear_acceleration = new GeometryMsgs.Vector3
                {
                    x = Physics.gravity.x + Random.Range(-imuNoise, imuNoise),
                    y = Physics.gravity.y + Random.Range(-imuNoise, imuNoise),
                    z = Physics.gravity.z + Random.Range(-imuNoise, imuNoise)
                }
            };

            // Publish to ROS
            ros.Publish(rosTopicPrefix + "/imu/data", imuMsg);

            yield return new WaitForSeconds(0.02f); // 50 Hz
        }
    }

    void OnValidate()
    {
        cameraWidth = Mathf.Clamp(cameraWidth, 16, 4096);
        cameraHeight = Mathf.Clamp(cameraHeight, 16, 4096);
        lidarRays = Mathf.Clamp(lidarRays, 1, 3600);
        lidarRange = Mathf.Clamp(lidarRange, 1f, 100f);
        lidarAngle = Mathf.Clamp(lidarAngle, 1f, 360f);
        imuNoise = Mathf.Clamp(imuNoise, 0f, 0.1f);
    }
}