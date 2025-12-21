---
sidebar_position: 1
title: Quick Start Guide
---

# Quick Start Guide: Physical AI & Humanoid Robotics

This quick start guide will help you get up and running with the Physical AI & Humanoid Robotics textbook and its associated code examples.

## Prerequisites

Before starting with the textbook, ensure you have the following installed:

- **Python 3.8 or higher**
- **ROS 2 (Humble Hawksbill or newer)**
- **Node.js 18 or higher** (for the documentation site)
- **Git**
- **Docker** (optional, for isolated environments)

## Setting Up Your Development Environment

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/physical-ai-textbook.git
cd physical-ai-textbook
```

### 2. Install Python Dependencies

```bash
pip install rclpy openai speechrecognition pyttsx3 webrtcvad pyaudio
```

### 3. Set Up ROS 2 Workspace

```bash
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

### 4. Install OpenAI API Key

Create a `.env` file in your project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Examples

### Basic ROS 2 Node Example

Navigate to the code examples for Module 1:

```bash
cd tutorials/code-examples/module-1/
python basic_publisher_subscriber.py
```

### Vision-Language-Action Example

For the VLA module examples:

```bash
cd tutorials/code-examples/module-4/
python voice_command_processor.py
```

## Understanding the Textbook Structure

The textbook is organized into four main modules:

1. **Module 1: The Robotic Nervous System (ROS 2)** - Covers the fundamentals of ROS 2, nodes, topics, and services
2. **Module 2: The Digital Twin (Gazebo & Unity)** - Focuses on simulation environments and digital replicas
3. **Module 3: The AI-Robot Brain (NVIDIA Isaac)** - Explores perception, navigation, and locomotion
4. **Module 4: Vision-Language-Action (VLA)** - Integrates voice, language, and action for humanoid robots

## Key Concepts Covered

- **Embodied AI**: AI systems that interact with the physical world through robotic bodies
- **Human-Robot Interaction**: Natural and intuitive ways for humans to communicate with robots
- **Perception Systems**: Computer vision, sensor fusion, and environmental understanding
- **Action Planning**: How robots decide what actions to take to achieve goals
- **Safety and Ethics**: Critical considerations for deploying robots in human environments

## Running the Documentation Site

To run the textbook documentation locally:

```bash
cd docusaurus-book
npm install
npm start
```

The documentation will be available at `http://localhost:3000`.

## Getting Help

- Check the **Reference** section for the glossary and index
- Review the **Exercises** at the end of each module
- Use the **Self-Check Questions** to validate your understanding
- Join our community forum for additional support

## Next Steps

After completing this quick start:

1. Begin with Module 1 to understand ROS 2 fundamentals
2. Progress through each module sequentially
3. Complete the exercises and projects
4. Try building your own humanoid robot applications

## Troubleshooting

### Common Issues

1. **ROS 2 Commands Not Found**: Make sure to source your ROS 2 installation:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **Python Import Errors**: Ensure you have installed all required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Microphone Access**: If using voice commands, ensure your system allows microphone access to the Python process.

4. **API Key Issues**: Verify your OpenAI API key is correctly set in your environment variables.

This quick start should have you up and running with the Physical AI & Humanoid Robotics textbook. For detailed information on each topic, proceed through the modules in order.