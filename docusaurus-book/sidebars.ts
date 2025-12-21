import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  textbook: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro'],
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2/intro',
        'module-1-ros2/ros-nodes-topics',
        'module-1-ros2/python-ros-control',
        'module-1-ros2/urdf-humanoids',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-digital-twin/gazebo-intro',
        'module-2-digital-twin/unity-simulation',
        'module-2-digital-twin/sensor-simulation',
        'module-2-digital-twin/human-robot-interaction',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-ai-brain/isaac-sim-overview',
        'module-3-ai-brain/vslam-navigation',
        'module-3-ai-brain/biped-locomotion',
        'module-3-ai-brain/perception-pipelines',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/whisper-voice-commands',
        'module-4-vla/llm-ros-integration',
        'module-4-vla/capstone-project',
        'module-4-vla/humanoid-interaction',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'quickstart',
        'glossary',
        'index',
        'accessibility',
      ],
    },
  ],
};

export default sidebars;
