#!/usr/bin/env python3

"""
Vision-Language-Action (VLA) Human-Robot Interaction

This script demonstrates a human-robot interaction interface using VLA models,
including speech recognition, visual perception, and action execution.
"""

import speech_recognition as sr
import pyttsx3
import cv2
import numpy as np
import threading
import time
from queue import Queue
import torch
import json
import os


class VLAInteractionInterface:
    """
    Human-robot interaction interface for Vision-Language-Action models
    """
    def __init__(self, vla_controller, camera_index=0):
        self.vla_controller = vla_controller

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume level

        # Initialize camera
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            print("Warning: Could not open camera. Using dummy images.")
            self.camera = None

        # Communication queues
        self.command_queue = Queue()
        self.response_queue = Queue()

        # Interaction state
        self.listening = False
        self.running = False
        self.interaction_history = []

        # Setup speech recognition
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        print("VLA Interaction Interface initialized.")

    def start_interaction(self):
        """Start the human-robot interaction interface"""
        self.listening = True
        self.running = True

        # Start the voice recognition thread
        recognition_thread = threading.Thread(target=self.voice_recognition_loop, daemon=True)
        recognition_thread.start()

        # Start the command processing thread
        processing_thread = threading.Thread(target=self.command_processing_loop, daemon=True)
        processing_thread.start()

        print("VLA Interaction Interface started. Listening for commands...")

        # Keep the interface running
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping interface...")
            self.stop()

    def voice_recognition_loop(self):
        """Continuously listen for voice commands"""
        print("Voice recognition thread started.")

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        while self.running:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)

                # Recognize speech
                command_text = self.recognizer.recognize_google(audio)
                print(f"Recognized: {command_text}")

                # Add command to queue
                self.command_queue.put({
                    'command': command_text,
                    'timestamp': time.time(),
                    'type': 'voice'
                })

            except sr.WaitTimeoutError:
                # This is normal - just continue listening
                continue
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")
            except Exception as e:
                print(f"Error in voice recognition: {e}")

            time.sleep(0.1)  # Small delay to prevent excessive CPU usage

    def command_processing_loop(self):
        """Process commands from the queue"""
        print("Command processing thread started.")

        while self.running:
            try:
                if not self.command_queue.empty():
                    command_data = self.command_queue.get(timeout=0.1)

                    # Get current image from camera
                    if self.camera:
                        ret, frame = self.camera.read()
                        if not ret:
                            print("Could not read from camera, using dummy image")
                            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    else:
                        # Use dummy image if camera is not available
                        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                    # Process the command with VLA model
                    try:
                        command = command_data['command']
                        print(f"Processing command: '{command}'")

                        # Execute the task using VLA model
                        actions = self.vla_controller.execute_task(frame, command)

                        # Store interaction in history
                        interaction = {
                            'command': command,
                            'actions': actions.tolist() if isinstance(actions, np.ndarray) else actions,
                            'timestamp': command_data['timestamp'],
                            'response_time': time.time() - command_data['timestamp']
                        }
                        self.interaction_history.append(interaction)

                        print(f"Actions generated: {actions}")

                        # Respond to user
                        self.speak_response(f"I will execute the task: {command}")

                        # Simulate action execution
                        self.execute_simulated_actions(actions, command)

                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error processing your command: {str(e)}"
                        print(error_msg)
                        self.speak_response(error_msg)

                time.sleep(0.1)

            except Exception as e:
                print(f"Error in command processing: {e}")
                time.sleep(0.1)

    def speak_response(self, text):
        """Speak a response to the user"""
        print(f"Speaking: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def execute_simulated_actions(self, actions, command):
        """Simulate action execution (in real implementation, this would control a robot)"""
        print(f"Simulating action execution for command: '{command}'")
        print(f"Actions: {actions}")

        # In a real implementation, this would send commands to a robot
        # For simulation, we'll just wait and print status
        action_duration = np.random.uniform(1, 3)  # Random duration for simulation
        time.sleep(action_duration)

        print(f"Action execution completed for: {command}")

    def add_command(self, command_text):
        """Add a command to the queue (for testing or direct input)"""
        self.command_queue.put({
            'command': command_text,
            'timestamp': time.time(),
            'type': 'text'
        })

    def get_interaction_history(self):
        """Get the history of interactions"""
        return self.interaction_history

    def save_interaction_history(self, filepath):
        """Save interaction history to a file"""
        with open(filepath, 'w') as f:
            json.dump(self.interaction_history, f, indent=2)
        print(f"Interaction history saved to {filepath}")

    def stop(self):
        """Stop the interaction interface"""
        print("Stopping VLA Interaction Interface...")
        self.running = False

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()
        print("VLA Interaction Interface stopped.")


class MockVLAController:
    """
    Mock VLA controller for demonstration purposes
    """
    def __init__(self):
        self.device = torch.device("cpu")  # Use CPU for demo
        print("Mock VLA Controller initialized.")

    def execute_task(self, image, natural_language_command):
        """
        Execute a task based on natural language command and visual input
        """
        print(f"Processing command: '{natural_language_command}' with image shape {image.shape}")

        # Simulate some processing time
        time.sleep(0.5)

        # Generate dummy actions based on command
        if "pick" in natural_language_command.lower() or "grasp" in natural_language_command.lower():
            # Simulate picking action (7-DOF robotic arm)
            actions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        elif "move" in natural_language_command.lower() or "go" in natural_language_command.lower():
            # Simulate movement action
            actions = np.array([0.2, 0.1, 0.0, 0.3, 0.4, 0.5, 0.6])
        elif "place" in natural_language_command.lower() or "put" in natural_language_command.lower():
            # Simulate placing action
            actions = np.array([0.3, 0.4, 0.1, 0.2, 0.6, 0.7, 0.5])
        else:
            # Default action
            actions = np.random.randn(7)

        # Add some variation based on image content (simulated)
        image_factor = np.mean(image) / 255.0
        actions = actions + (image_factor * 0.1)

        return actions

    def get_similarity_score(self, image, text):
        """
        Get similarity score between image and text (for evaluation)
        """
        # Simulate similarity based on text content
        if any(word in text.lower() for word in ["red", "blue", "green", "object", "cup", "box"]):
            return np.random.uniform(0.7, 0.9)  # High similarity for object-related commands
        else:
            return np.random.uniform(0.3, 0.6)  # Lower similarity otherwise


def main():
    """Main function to demonstrate VLA interaction"""
    print("Initializing VLA Human-Robot Interaction Interface...")

    # Initialize mock VLA controller (in practice, this would be a trained model)
    vla_controller = MockVLAController()

    # Initialize interaction interface
    interface = VLAInteractionInterface(vla_controller)

    print("\nVLA Interaction Demo")
    print("=" * 40)
    print("Commands to try:")
    print("- 'Pick up the red cup'")
    print("- 'Move the box to the left'")
    print("- 'Place the object on the table'")
    print("- 'Open the door'")
    print("\nPress Ctrl+C to stop the interface")

    try:
        # Start the interface
        interface.start_interaction()

    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Ensure cleanup
        interface.stop()

        # Save interaction history
        if interface.get_interaction_history():
            interface.save_interaction_history("vla_interaction_history.json")

        print("Demo completed!")


if __name__ == "__main__":
    main()