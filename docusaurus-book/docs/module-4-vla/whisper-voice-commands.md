---
title: Whisper Voice Commands
sidebar_label: Whisper Voice Commands
description: Implementing voice command processing with OpenAI Whisper for humanoid robots
sidebar_position: 1
---

# Whisper Voice Commands

## Overview

This section covers implementing voice command processing using OpenAI Whisper for humanoid robots. Whisper is a state-of-the-art speech recognition model that can transcribe speech to text with high accuracy. When integrated with robotics systems, it enables natural human-robot interaction through voice commands.

## Learning Objectives

- Understand the architecture of voice command processing systems
- Implement real-time speech recognition using Whisper
- Integrate voice processing with robot action execution
- Handle voice command validation and error recovery
- Design conversational flows for humanoid interaction

## Architecture Overview

The voice command processing system consists of several key components:

1. **Audio Input**: Captures voice commands from the user
2. **Speech Recognition**: Transcribes speech to text using Whisper
3. **Intent Processing**: Interprets the meaning of the command
4. **Action Mapping**: Translates commands to robot actions
5. **Execution Layer**: Executes the robot actions
6. **Feedback System**: Provides confirmation and status updates

## Implementation Pattern

### Basic Voice Command Processor

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from openai import OpenAI
import speech_recognition as sr
import json
import threading
import pyaudio

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

        # Audio processing parameters
        self.recognizer.energy_threshold = 300  # Adjust for ambient noise
        self.recognizer.dynamic_energy_threshold = True

        # Command history for context
        self.command_history = []

        # Start voice processing
        self.get_logger().info('Voice Command Processor Node Started')

    def process_voice_command(self):
        """Process a single voice command"""
        try:
            with self.microphone as source:
                self.get_logger().info("Listening for voice command...")
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=10.0)

            # Transcribe audio using Whisper
            transcription = self.transcribe_audio(audio)
            if transcription:
                self.get_logger().info(f"Transcribed: {transcription}")

                # Process the command and generate robot action
                robot_action = self.process_command(transcription)

                if robot_action:
                    # Publish the action
                    command_msg = String()
                    command_msg.data = json.dumps(robot_action)
                    self.command_pub.publish(command_msg)
                    self.get_logger().info(f"Published robot action: {robot_action}")

                    # Add to command history
                    self.command_history.append({
                        'command': transcription,
                        'action': robot_action,
                        'timestamp': self.get_clock().now().seconds_nanoseconds()
                    })

                return True
            else:
                self.get_logger().warn("Could not transcribe audio")
                return False

        except sr.WaitTimeoutError:
            self.get_logger().info("No speech detected within timeout period")
            return False
        except Exception as e:
            self.get_logger().error(f"Error processing voice command: {e}")
            return False

    def transcribe_audio(self, audio):
        """Transcribe audio using Whisper API"""
        try:
            # Save audio to temporary file for Whisper API
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Convert audio to WAV format
                wav_data = audio.get_wav_data()
                temp_file.write(wav_data)
                temp_file_path = temp_file.name

            try:
                with open(temp_file_path, 'rb') as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                return transcript.text
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)

        except Exception as e:
            self.get_logger().error(f"Whisper transcription error: {e}")
            return None

    def process_command(self, command_text):
        """Process the transcribed command and generate robot action"""
        # In a real implementation, this would use an LLM to interpret the command
        # For this example, we'll use simple keyword matching
        command_text = command_text.lower().strip()

        # Simple command mapping (in practice, use LLM for more complex parsing)
        if 'move' in command_text or 'go to' in command_text:
            if 'kitchen' in command_text:
                return {
                    'action': 'navigate_to',
                    'parameters': {'location': 'kitchen'}
                }
            elif 'living room' in command_text or 'livingroom' in command_text:
                return {
                    'action': 'navigate_to',
                    'parameters': {'location': 'living_room'}
                }
        elif 'pick up' in command_text or 'get' in command_text:
            # Extract object name using simple parsing
            object_name = self.extract_object_name(command_text)
            return {
                'action': 'pick_up',
                'parameters': {'object': object_name}
            }
        elif 'place' in command_text or 'put' in command_text:
            location = self.extract_location_name(command_text)
            return {
                'action': 'place',
                'parameters': {'location': location}
            }
        elif 'stop' in command_text or 'halt' in command_text:
            return {
                'action': 'stop',
                'parameters': {}
            }
        elif 'hello' in command_text or 'hi' in command_text:
            return {
                'action': 'speak',
                'parameters': {'text': 'Hello! How can I help you today?'}
            }

        # If no specific action is identified, return None
        return None

    def extract_object_name(self, command_text):
        """Extract object name from command using simple parsing"""
        # This is a simplified approach - in practice, use NLP techniques
        # Look for common object words in the command
        common_objects = ['cup', 'bottle', 'book', 'ball', 'box', 'plate', 'fork', 'spoon']
        for obj in common_objects:
            if obj in command_text:
                return obj
        return 'object'

    def extract_location_name(self, command_text):
        """Extract location name from command using simple parsing"""
        common_locations = ['kitchen', 'living room', 'bedroom', 'office', 'dining room']
        for loc in common_locations:
            if loc in command_text:
                return loc.replace(' ', '_')
        return 'location'

def main(args=None):
    rclpy.init(args=args)
    voice_processor = VoiceCommandProcessor()

    # Process commands in a loop
    try:
        while rclpy.ok():
            voice_processor.process_voice_command()
            # Small delay to prevent excessive CPU usage
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        voice_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Voice Activity Detection

For more robust voice command processing, implement voice activity detection to distinguish between speech and silence:

```python
import webrtcvad
import collections
import pyaudio
import logging

class VoiceActivityDetector:
    def __init__(self, aggressiveness=3):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.ring_buffer = collections.deque(maxlen=300)
        self.triggered = False
        self.vad_frames = []
        self.rate = 16000  # Sample rate
        self.frame_duration = 30  # Frame duration in ms
        self.frame_size = int(self.rate * self.frame_duration / 1000)

    def is_speech(self, frame):
        """Check if the audio frame contains speech"""
        try:
            return self.vad.is_speech(frame, self.rate)
        except:
            return False

    def process_audio_chunk(self, audio_chunk):
        """Process an audio chunk for voice activity"""
        is_speech = self.is_speech(audio_chunk)
        self.ring_buffer.append((audio_chunk, is_speech))

        if not self.triggered:
            # Voice activity not yet detected
            num_voiced = len([f for f, speech in self.ring_buffer if speech])
            if num_voiced > 0.5 * self.ring_buffer.maxlen:
                self.triggered = True
                # Clear previous frames before voice activity
                self.vad_frames = [f for f, s in self.ring_buffer]
        else:
            # Voice activity detected, waiting for silence
            self.vad_frames.append(audio_chunk)
            num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
            if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                # End of speech detected
                self.triggered = False
                # Return the collected voice frames
                frames = self.vad_frames
                self.vad_frames = []
                return frames

        return None
```

## Voice Command Validation

It's important to validate voice commands before execution to ensure safety and correctness:

```python
class CommandValidator:
    def __init__(self, robot_capabilities, environment_map):
        self.capabilities = robot_capabilities
        self.environment_map = environment_map

    def validate_voice_command(self, command_text, action):
        """Validate voice command for safety and feasibility"""
        # Check if action is supported by the robot
        if action['action'] not in self.capabilities.get('supported_actions', []):
            return False, f"Action '{action['action']}' is not supported by this robot"

        # Validate parameters based on action type
        if action['action'] == 'navigate_to':
            location = action['parameters'].get('location')
            if not location:
                return False, "Navigation command missing location parameter"

            if location not in self.environment_map.get('locations', []):
                return False, f"Location '{location}' does not exist in the environment"

        elif action['action'] == 'pick_up':
            obj_name = action['parameters'].get('object')
            if not obj_name:
                return False, "Pick-up command missing object parameter"

        # Check for potential safety issues
        if self.would_cause_collision(action):
            return False, "Command would cause collision"

        return True, "Command is valid"

    def would_cause_collision(self, action):
        """Check if the action would cause a collision"""
        # Implementation depends on robot and environment
        # This is a simplified example
        return False
```

## Error Handling and Recovery

Implement proper error handling for voice recognition failures:

```python
class VoiceCommandErrorHandling:
    def __init__(self, node):
        self.node = node
        self.retry_count = 0
        self.max_retries = 3

    def handle_recognition_error(self, error_type, error_message):
        """Handle different types of recognition errors"""
        if error_type == "no_speech":
            self.node.speak("I didn't hear any speech. Please speak clearly and try again.")
        elif error_type == "recognition_failed":
            self.node.speak("I couldn't understand your command. Could you please repeat it?")
        elif error_type == "service_error":
            self.node.speak("There was an error processing your command. Please try again later.")
        elif error_type == "timeout":
            self.node.speak("I didn't receive a command in time. Please try again.")

    def handle_action_error(self, action, error):
        """Handle errors when executing robot actions"""
        self.node.get_logger().error(f"Error executing action {action}: {error}")
        self.node.speak(f"I encountered an error executing your command: {str(error)}")
```

## Self-Check Questions


## Summary

This section covered implementing voice command processing with OpenAI Whisper for humanoid robots. We explored the architecture of voice command systems, implementation patterns for real-time processing, and important considerations for validation and error handling. The integration of Whisper with robotics systems enables natural and intuitive human-robot interaction.