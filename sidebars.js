// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        {
          type: 'doc',
          id: 'book/physical-ai',
        },
        {
          type: 'doc',
          id: 'book/ros2',
        },
        {
          type: 'doc',
          id: 'book/digital-twin',
        },
        {
          type: 'doc',
          id: 'book/isaac',
        },
        {
          type: 'doc',
          id: 'book/vla',
        },
        {
          type: 'doc',
          id: 'book/conversational-robotics',
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 - Robotic Nervous System',
      items: [
        {
          type: 'doc',
          id: 'chapter-1/index',
        },
        {
          type: 'doc',
          id: 'chapter-2/index',
        },
        {
          type: 'doc',
          id: 'chapter-3/index',
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin - Gazebo + Unity',
      items: [
        {
          type: 'doc',
          id: 'chapter-4/index',
        },
        {
          type: 'doc',
          id: 'chapter-5/index',
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac - Perception, VSLAM, Nav2',
      items: [
        {
          type: 'doc',
          id: 'chapter-6/index',
        },
        {
          type: 'doc',
          id: 'chapter-7/index',
        },
        {
          type: 'doc',
          id: 'chapter-8/index',
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action - Whisper → LLM → ROS',
      items: [
        {
          type: 'doc',
          id: 'chapter-9/index',
        },
        {
          type: 'doc',
          id: 'chapter-10/index',
        },
        {
          type: 'doc',
          id: 'chapter-11/index',
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 5: Capstone Project',
      items: [
        {
          type: 'doc',
          id: 'chapter-12/index',
        },
      ],
    },
  ],
};

export default sidebars;