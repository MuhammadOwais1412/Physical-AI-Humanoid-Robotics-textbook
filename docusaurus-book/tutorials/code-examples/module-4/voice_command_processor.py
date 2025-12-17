# VLA Integration Example
# voice_command_processor.py

import whisper
import openai
import json
from typing import Dict, Any, Optional

class VoiceCommandProcessor:
    """Process voice commands using Whisper and translate to robot actions"""
    
    def __init__(self, whisper_model="base", openai_api_key=None):
        self.whisper_model = whisper.load_model(whisper_model)
        if openai_api_key:
            openai.api_key = openai_api_key
        self.command_history = []
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio file to text using Whisper"""
        result = self.whisper_model.transcribe(audio_file_path)
        return result['text'].strip()
    
    def process_voice_command(self, audio_file_path: str) -> Dict[str, Any]:
        """Process voice command and return structured robot action"""
        # Transcribe the audio
        transcription = self.transcribe_audio(audio_file_path)
        
        # Use LLM to convert natural language to structured command
        structured_command = self.convert_to_structured_command(transcription)
        
        # Add to history
        self.command_history.append({
            'transcription': transcription,
            'structured_command': structured_command,
            'timestamp': self.get_current_timestamp()
        })
        
        return structured_command
    
    def convert_to_structured_command(self, natural_language: str) -> Dict[str, Any]:
        """Use LLM to convert natural language to structured command"""
        prompt = f"""
        Convert the following natural language command to a structured robot command.
        
        Natural Language: {natural_language}
        
        Return a JSON object with this structure:
        {{
            "action": "navigate_to | pick_up | place | speak | detect_object | ...",
            "parameters": {{
                "location": "optional location",
                "object": "optional object",
                "text": "optional text",
                "confidence": 0.0-1.0
            }},
            "intent": "user's intent",
            "entities": ["extracted entities"]
        }}
        
        JSON:
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message['content'].strip()
            
            # Extract JSON if wrapped in code blocks
            if content.startswith('')]
            elif content.startswith('')]
            
            return json.loads(content)
        
        except Exception as e:
            print(f"Error converting to structured command: {e}")
            return {
                "action": "unknown",
                "parameters": {},
                "intent": "unknown",
                "entities": []
            }
    
    def get_current_timestamp(self):
        import time
        return time.time()

# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = VoiceCommandProcessor(openai_api_key="YOUR_API_KEY")
    
    # Example: Process a voice command
    # command = processor.process_voice_command("path/to/audio.wav")
    # print(f"Processed command: {command}")
    
    print("VoiceCommandProcessor initialized successfully")

