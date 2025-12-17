# Solutions for Module 4: Vision-Language-Action (VLA)

## Exercise 1: Voice Command Processing
**Problem**: Implement a voice command processor that uses Whisper to transcribe user commands and translates them to robot actions.

**Solution**:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from openai import OpenAI
import speech_recognition as sr
import json
import threading
import queue

class VoiceCommandProcessor(Node):
    def __init__(self):
        super().__init__('voice_command_processor')

        # Publisher for processed commands
        self.command_pub = self.create_publisher(String, 'processed_commands', 10)

        # Initialize Whisper client
        self.client = OpenAI()  # Assumes OpenAI API key is set in environment

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Setup for continuous listening
        self.command_queue = queue.Queue()
        self.listening = True

        # Start listening thread
        self.listen_thread = threading.Thread(target=self.listen_continuously)
        self.listen_thread.start()

        # Timer to process commands
        self.timer = self.create_timer(0.1, self.process_commands)

    def listen_continuously(self):
        """Continuously listen for voice commands"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        while self.listening:
            try:
                with self.microphone as source:
                    self.get_logger().info("Listening for voice command...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

                # Send audio to Whisper for transcription
                transcription = self.transcribe_audio(audio)
                if transcription:
                    self.command_queue.put(transcription)

            except sr.WaitTimeoutError:
                continue  # Continue listening
            except Exception as e:
                self.get_logger().error(f"Error in voice recognition: {e}")
                continue

    def transcribe_audio(self, audio):
        """Transcribe audio using Whisper API"""
        try:
            # Save audio to temporary file for Whisper API
            audio_data = audio.get_wav_data()
            with open('/tmp/temp_audio.wav', 'wb') as f:
                f.write(audio_data)

            with open('/tmp/temp_audio.wav', 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            return transcript.text
        except Exception as e:
            self.get_logger().error(f"Transcription error: {e}")
            return None

    def process_commands(self):
        """Process commands from the queue"""
        while not self.command_queue.empty():
            try:
                user_input = self.command_queue.get_nowait()
                self.get_logger().info(f"Processing command: {user_input}")

                # Process with LLM to extract intent and parameters
                ros_command = self.llm_process_input(user_input)

                if ros_command:
                    # Publish the ROS command
                    command_msg = String()
                    command_msg.data = json.dumps(ros_command)
                    self.command_pub.publish(command_msg)
                    self.get_logger().info(f'Published ROS command: {ros_command}')

            except queue.Empty:
                break

    def llm_process_input(self, user_input):
        """Use LLM to process input and generate ROS command"""
        prompt = f"""
        Convert the following natural language command to a ROS command structure.
        The output should be a JSON object with the following format:
        {{
            "action": "move_to | pick_up | place | navigate | ...",
            "parameters": {{
                "x": number,
                "y": number,
                "z": number,
                "object": "object_name",
                "location": "location_name"
            }}
        }}

        Input: {user_input}

        Output JSON:
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            # Extract and parse the JSON response
            result = response.choices[0].message.content.strip()

            # Clean up the response to extract JSON
            if result.startswith('```json'):
                result = result[7:result.rfind('```')]
            elif result.startswith('```'):
                result = result[3:result.rfind('```')]

            return json.loads(result)

        except Exception as e:
            self.get_logger().error(f'LLM processing error: {e}')
            return None

    def destroy_node(self):
        self.listening = False
        if self.listen_thread.is_alive():
            self.listen_thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    voice_processor = VoiceCommandProcessor()

    try:
        rclpy.spin(voice_processor)
    except KeyboardInterrupt:
        pass
    finally:
        voice_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise 2: LLM-ROS Integration
**Problem**: Create a bridge between LLMs and ROS that can translate natural language to ROS actions.

**Solution**:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from openai import OpenAI
import json
import asyncio

class LLMROSBridge(Node):
    def __init__(self):
        super().__init__('llm_ros_bridge')

        # Publisher for commands to other nodes
        self.command_publisher = self.create_publisher(String, 'natural_language_commands', 10)

        # Subscriber for user input (could be from voice processor or UI)
        self.user_input_subscriber = self.create_subscription(
            String,
            'user_input',
            self.user_input_callback,
            10
        )

        # ROS command publisher
        self.ros_command_publisher = self.create_publisher(String, 'ros_commands', 10)

        # Initialize OpenAI client
        self.client = OpenAI()

        self.get_logger().info('LLM-ROS Bridge Node Started')

    def user_input_callback(self, msg):
        """Process user input and translate to ROS commands"""
        user_input = msg.data
        self.get_logger().info(f'Received user input: {user_input}')

        # Process with LLM to extract intent and parameters
        ros_command = self.llm_process_input(user_input)

        if ros_command:
            # Publish the ROS command
            command_msg = String()
            command_msg.data = json.dumps(ros_command)
            self.ros_command_publisher.publish(command_msg)
            self.get_logger().info(f'Published ROS command: {ros_command}')

    def llm_process_input(self, user_input):
        """Use LLM to process input and generate ROS command"""
        prompt = f"""
        Convert the following natural language command to a ROS command structure.
        The output should be a JSON object with the following format:
        {{
            "action": "move_to | pick_up | place | navigate | follow | stop | speak",
            "parameters": {{
                "x": number,
                "y": number,
                "z": number,
                "object": "object_name",
                "location": "location_name",
                "text": "text_to_speak"
            }}
        }}

        Consider these safety constraints:
        - Validate that locations exist in the environment
        - Check that requested actions are safe to perform
        - Verify that objects exist before manipulation

        Input: {user_input}

        Output JSON:
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            # Extract and parse the JSON response
            result = response.choices[0].message.content.strip()

            # Clean up the response to extract JSON
            if result.startswith('```json'):
                result = result[7:result.rfind('```')]
            elif result.startswith('```'):
                result = result[3:result.rfind('```')]

            return json.loads(result)

        except Exception as e:
            self.get_logger().error(f'LLM processing error: {e}')
            return None

class LLMTaskPlanner(Node):
    def __init__(self):
        super().__init__('llm_task_planner')

        # Publisher for task plans
        self.plan_publisher = self.create_publisher(String, 'task_plans', 10)

        # Service client for environment state
        self.state_client = self.create_client(String, 'get_environment_state')

        self.get_logger().info('LLM Task Planner Node Started')

    def generate_task_plan(self, goal_description):
        """Generate a task plan using LLM based on the goal"""
        # Get current environment state
        state_request = String()
        state_request.data = "current_state"

        if self.state_client.wait_for_service(timeout_sec=1.0):
            future = self.state_client.call_async(state_request)
            # Wait for response and get current state
            # For simplicity, assuming we get the state synchronously

        prompt = f"""
        Given the current environment state and a goal, create a task plan.
        The environment contains: [describe environment]
        The goal is: {goal_description}

        Create a plan with these constraints:
        - Each step should be a simple ROS action
        - Consider object locations, robot capabilities, and obstacles
        - Handle potential failures in each step
        - Include safety checks between steps

        Return the plan as a JSON array of steps:
        [
            {{
                "step": 1,
                "action": "navigate_to",
                "parameters": {{"location": "kitchen"}},
                "success_condition": "robot_at_location",
                "failure_recovery": "retry_with_different_path"
            }},
            {{
                "step": 2,
                "action": "detect_object",
                "parameters": {{"object": "cup"}},
                "success_condition": "object_detected",
                "failure_recovery": "search_in_adjacent_room"
            }}
        ]
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            plan_text = response.choices[0].message.content.strip()
            if plan_text.startswith('```json'):
                plan_text = plan_text[7:plan_text.rfind('```')]
            elif plan_text.startswith('```'):
                plan_text = plan_text[3:plan_text.rfind('```')]

            return json.loads(plan_text)

        except Exception as e:
            self.get_logger().error(f'Task planning error: {e}')
            return None

    def execute_plan(self, plan):
        """Execute the generated plan step by step"""
        for step in plan:
            success = self.execute_step(step)
            if not success:
                self.get_logger().error(f'Plan execution failed at step: {step}')
                return False
        return True

    def execute_step(self, step):
        """Execute a single step of the plan"""
        action = step['action']
        params = step['parameters']

        # Map action to ROS service call or action
        if action == 'navigate_to':
            return self.navigate_to_location(params['location'])
        elif action == 'detect_object':
            return self.detect_object(params['object'])
        # ... other actions

        # Check success condition and handle failure if needed
        return True

def main(args=None):
    rclpy.init(args=args)
    llm_bridge = LLMROSBridge()
    task_planner = LLMTaskPlanner()

    try:
        # Create a MultiThreadedExecutor to run both nodes
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(llm_bridge)
        executor.add_node(task_planner)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        llm_bridge.destroy_node()
        task_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise 3: Conversational Humanoid
**Problem**: Integrate voice processing, LLM integration, and ROS control into a complete conversational humanoid system.

**Solution**:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from openai import OpenAI
import json
import speech_recognition as sr
import pyttsx3
import threading
import queue
from enum import Enum

class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    EXECUTING = "executing"

class ConversationalHumanoid(Node):
    def __init__(self):
        super().__init__('conversational_humanoid')

        # Publishers
        self.speech_pub = self.create_publisher(String, 'tts_commands', 10)
        self.movement_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.nav_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Initialize components
        self.client = OpenAI()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume level

        # Conversation state
        self.conversation_state = ConversationState.IDLE
        self.context = []  # Store conversation history

        # Setup for continuous listening
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.listening = True

        # Start listening thread
        self.listen_thread = threading.Thread(target=self.listen_continuously)
        self.listen_thread.start()

        # Timer to process commands
        self.timer = self.create_timer(0.1, self.process_commands)

        self.get_logger().info('Conversational Humanoid Node Started')

    def listen_continuously(self):
        """Continuously listen for voice commands"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        while self.listening:
            try:
                with self.microphone as source:
                    self.get_logger().info("Listening...")
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=10)

                # Send audio to Whisper for transcription
                transcription = self.transcribe_audio(audio)
                if transcription:
                    self.command_queue.put(transcription)
                    self.conversation_state = ConversationState.PROCESSING

            except sr.WaitTimeoutError:
                continue  # Continue listening
            except Exception as e:
                self.get_logger().error(f"Error in voice recognition: {e}")
                continue

    def transcribe_audio(self, audio):
        """Transcribe audio using Whisper API"""
        try:
            # Save audio to temporary file for Whisper API
            audio_data = audio.get_wav_data()
            with open('/tmp/temp_audio.wav', 'wb') as f:
                f.write(audio_data)

            with open('/tmp/temp_audio.wav', 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            return transcript.text
        except Exception as e:
            self.get_logger().error(f"Transcription error: {e}")
            return None

    def process_commands(self):
        """Process commands from the queue"""
        while not self.command_queue.empty():
            try:
                user_input = self.command_queue.get_nowait()
                self.get_logger().info(f"Processing: {user_input}")

                # Add to context
                self.context.append({"role": "user", "content": user_input})

                # Generate response using LLM with context
                response = self.generate_response_with_context(user_input)

                if response:
                    self.response_queue.put(response)
                    self.conversation_state = ConversationState.SPEAKING
                    self.speak_response(response)

            except queue.Empty:
                break

    def generate_response_with_context(self, user_input):
        """Generate response using LLM with conversation context"""
        # Prepare context for the LLM
        messages = [
            {"role": "system", "content": "You are a helpful humanoid robot. Respond naturally and include appropriate actions when needed. Keep responses concise but informative."}
        ]

        # Add conversation history
        messages.extend(self.context[-10:])  # Use last 10 exchanges as context

        # Add the current user input
        messages.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7
            )

            response_text = response.choices[0].message.content.strip()

            # Add to context
            self.context.append({"role": "assistant", "content": response_text})

            # Check if response contains action commands
            action_commands = self.extract_action_commands(response_text)
            if action_commands:
                self.execute_actions(action_commands)

            return response_text

        except Exception as e:
            self.get_logger().error(f'LLM processing error: {e}')
            return "Sorry, I encountered an error processing your request."

    def extract_action_commands(self, text):
        """Extract action commands from the response text"""
        # This is a simplified approach - in practice, you'd use more sophisticated parsing
        action_prompt = f"""
        Extract action commands from this text. Return as JSON array with possible actions:
        ["speak", "gesture", "navigate", "pick_up", "place"]

        Text: {text}

        JSON array of actions:
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": action_prompt}],
                temperature=0.1
            )

            result = response.choices[0].message.content.strip()
            if result.startswith('```json'):
                result = result[7:result.rfind('```')]
            elif result.startswith('```'):
                result = result[3:result.rfind('```')]

            return json.loads(result)
        except:
            return []

    def execute_actions(self, actions):
        """Execute the extracted actions"""
        for action in actions:
            if action == "speak":
                # Already handled by speak_response
                pass
            elif action == "gesture":
                # Execute gesture commands
                self.execute_gesture()
            elif action == "navigate":
                # Execute navigation commands
                self.execute_navigation()
            # Add more action types as needed

    def speak_response(self, response):
        """Speak the response using TTS"""
        try:
            # Publish to TTS system (if using external TTS node)
            tts_msg = String()
            tts_msg.data = response
            self.speech_pub.publish(tts_msg)

            # Also speak directly using local TTS engine
            self.tts_engine.say(response)
            self.tts_engine.runAndWait()

            self.conversation_state = ConversationState.IDLE
        except Exception as e:
            self.get_logger().error(f"TTS error: {e}")

    def execute_gesture(self):
        """Execute gesture commands"""
        # Example: Move head or arms in a greeting gesture
        joint_msg = JointState()
        joint_msg.name = ['head_pan', 'head_tilt']
        joint_msg.position = [0.0, 0.2]  # Look up slightly
        self.movement_pub.publish(joint_msg)

    def execute_navigation(self):
        """Execute navigation commands"""
        # Example: Move forward slightly
        twist_msg = Twist()
        twist_msg.linear.x = 0.2  # Move forward at 0.2 m/s
        self.nav_pub.publish(twist_msg)

    def destroy_node(self):
        self.listening = False
        if self.listen_thread.is_alive():
            self.listen_thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    humanoid = ConversationalHumanoid()

    try:
        rclpy.spin(humanoid)
    except KeyboardInterrupt:
        pass
    finally:
        humanoid.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise 4: Capstone Project - Autonomous Assistant
**Problem**: Build a complete autonomous humanoid assistant that can perform complex tasks based on voice commands.

**Solution**:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image, LaserScan
from openai import OpenAI
import json
import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
import numpy as np

class AutonomousAssistant(Node):
    def __init__(self):
        super().__init__('autonomous_assistant')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, 'tts_commands', 10)
        self.status_pub = self.create_publisher(String, 'assistant_status', 10)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )
        self.camera_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.camera_callback, 10
        )

        # Initialize components
        self.client = OpenAI()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)

        # Robot state
        self.current_pose = Pose()
        self.obstacles = []
        self.objects_detected = []
        self.battery_level = 100.0

        # Task execution
        self.current_task = None
        self.task_queue = queue.Queue()
        self.is_executing = False

        # Setup for continuous operation
        self.command_queue = queue.Queue()
        self.listening = True

        # Start listening thread
        self.listen_thread = threading.Thread(target=self.listen_continuously)
        self.listen_thread.start()

        # Timer for task execution
        self.task_timer = self.create_timer(0.1, self.execute_current_task)

        # Timer for system monitoring
        self.monitor_timer = self.create_timer(1.0, self.system_monitor)

        self.get_logger().info('Autonomous Assistant Node Started')

    def listen_continuously(self):
        """Continuously listen for voice commands"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        while self.listening:
            try:
                with self.microphone as source:
                    self.get_logger().info("Listening for commands...")
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=15)

                transcription = self.transcribe_audio(audio)
                if transcription:
                    self.command_queue.put(transcription)

            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                self.get_logger().error(f"Error in voice recognition: {e}")
                continue

    def transcribe_audio(self, audio):
        """Transcribe audio using Whisper API"""
        try:
            audio_data = audio.get_wav_data()
            with open('/tmp/temp_audio.wav', 'wb') as f:
                f.write(audio_data)

            with open('/tmp/temp_audio.wav', 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            return transcript.text
        except Exception as e:
            self.get_logger().error(f"Transcription error: {e}")
            return None

    def laser_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        self.obstacles = []
        for i, range_val in enumerate(msg.ranges):
            if 0.1 < range_val < 5.0:  # Valid range readings
                angle = msg.angle_min + i * msg.angle_increment
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                self.obstacles.append((x, y, range_val))

    def camera_callback(self, msg):
        """Process camera data for object detection"""
        # In a real implementation, this would run object detection
        # For this example, we'll simulate object detection
        pass

    def system_monitor(self):
        """Monitor system status and update battery level"""
        # Simulate battery drain
        self.battery_level = max(0.0, self.battery_level - 0.01)

        # Check if battery is low
        if self.battery_level < 20.0:
            self.speak("Warning: Battery level is low. Returning to charging station.")
            # Add return to charging task to queue
            charging_task = {
                "action": "navigate_to",
                "parameters": {"location": "charging_station"},
                "priority": "high"
            }
            self.task_queue.put(charging_task)

    def process_voice_command(self, command):
        """Process voice command and generate task plan"""
        # Get environment context
        env_context = self.get_environment_context()

        prompt = f"""
        Given the current environment context and a user command, create a task plan.

        Environment context:
        - Current pose: {self.current_pose}
        - Obstacles detected: {len(self.obstacles)}
        - Battery level: {self.battery_level}%
        - Objects detected: {self.objects_detected}

        User command: {command}

        Create a detailed task plan as a JSON array with steps:
        [
            {{
                "step": 1,
                "action": "navigate_to | detect_object | pick_up | place | speak",
                "parameters": {{"location": "kitchen", "object": "cup"}},
                "success_condition": "robot_at_location | object_detected",
                "failure_recovery": "alternative_approach"
            }}
        ]
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a task planner for an autonomous humanoid robot. Create detailed, executable task plans that consider safety, environment, and robot capabilities."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            plan_text = response.choices[0].message.content.strip()
            if plan_text.startswith('```json'):
                plan_text = plan_text[7:plan_text.rfind('```')]
            elif plan_text.startswith('```'):
                plan_text = plan_text[3:plan_text.rfind('```')]

            task_plan = json.loads(plan_text)

            # Add tasks to queue
            for task in task_plan:
                self.task_queue.put(task)

            return True
        except Exception as e:
            self.get_logger().error(f'Task planning error: {e}')
            return False

    def get_environment_context(self):
        """Get current environment context"""
        return {
            "pose": {"x": self.current_pose.position.x, "y": self.current_pose.position.y},
            "obstacles": len(self.obstacles),
            "battery": self.battery_level,
            "objects": self.objects_detected
        }

    def execute_current_task(self):
        """Execute the current task in the queue"""
        if not self.is_executing and not self.task_queue.empty():
            try:
                self.current_task = self.task_queue.get_nowait()
                self.is_executing = True

                success = self.execute_task_step(self.current_task)
                if success:
                    self.get_logger().info(f"Completed task: {self.current_task['action']}")
                else:
                    self.get_logger().error(f"Failed task: {self.current_task['action']}")
                    # Add failure recovery here

                self.is_executing = False
                self.current_task = None
            except queue.Empty:
                pass

    def execute_task_step(self, task):
        """Execute a single task step"""
        action = task['action']
        params = task.get('parameters', {})

        try:
            if action == 'navigate_to':
                return self.navigate_to_location(params.get('location'))
            elif action == 'detect_object':
                return self.detect_object(params.get('object'))
            elif action == 'pick_up':
                return self.pick_up_object(params.get('object'))
            elif action == 'place':
                return self.place_object(params.get('location'))
            elif action == 'speak':
                self.speak(params.get('text', 'Hello'))
                return True
            else:
                self.get_logger().error(f"Unknown action: {action}")
                return False
        except Exception as e:
            self.get_logger().error(f"Error executing task {action}: {e}")
            return False

    def navigate_to_location(self, location):
        """Navigate to a specified location"""
        # In a real implementation, this would use navigation stack
        # For this example, we'll simulate navigation
        self.speak(f"Navigating to {location}")

        # Publish velocity commands to move
        twist = Twist()
        twist.linear.x = 0.3  # Move forward
        for _ in range(10):  # Simulate moving for 1 second
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)

        # Stop
        twist.linear.x = 0.0
        self.cmd_vel_pub.publish(twist)

        return True

    def detect_object(self, obj_name):
        """Detect a specific object"""
        self.speak(f"Looking for {obj_name}")
        # Simulate object detection
        # In real implementation, this would use computer vision
        return True

    def pick_up_object(self, obj_name):
        """Pick up an object"""
        self.speak(f"Picking up {obj_name}")
        # Simulate pick-up action
        # In real implementation, this would control manipulator
        return True

    def place_object(self, location):
        """Place object at location"""
        self.speak(f"Placing object at {location}")
        # Simulate place action
        # In real implementation, this would control manipulator
        return True

    def speak(self, text):
        """Speak text using TTS"""
        try:
            # Publish to TTS system
            tts_msg = String()
            tts_msg.data = text
            self.speech_pub.publish(tts_msg)

            # Also speak locally
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

            return True
        except Exception as e:
            self.get_logger().error(f"TTS error: {e}")
            return False

    def destroy_node(self):
        self.listening = False
        if self.listen_thread.is_alive():
            self.listen_thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    assistant = AutonomousAssistant()

    try:
        # Process any voice commands in the queue
        while rclpy.ok():
            if not assistant.command_queue.empty():
                try:
                    command = assistant.command_queue.get_nowait()
                    assistant.get_logger().info(f"Processing command: {command}")
                    assistant.process_voice_command(command)
                except queue.Empty:
                    pass

            rclpy.spin_once(assistant, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        assistant.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```