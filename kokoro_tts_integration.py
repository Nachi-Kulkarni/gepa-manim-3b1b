#!/usr/bin/env python3
"""
Kokoro-82M Integration for Video Pipeline
Uses the newer Kokoro-82M model to generate narration audio from transcript
"""

import torch
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
import os

class Kokoro82MGenerator:
    """Kokoro-82M audio generation using the newer model"""
    
    def __init__(self):
        self.model = None
        self.pipelines = {}
        self.setup_kokoro()
    
    def setup_kokoro(self):
        """Initialize Kokoro-82M model and pipelines"""
        try:
            # Import Kokoro modules
            import kokoro
            
            print("🎤 Initializing Kokoro-82M...")
            
            # Load the 82M model
            self.model = kokoro.KModel(repo_id='hexgrad/Kokoro-82M').to('cpu').eval()
            
            # Initialize pipeline for American English (primary)
            self.pipeline = kokoro.KPipeline(lang_code='a', model=False)
            
            # Set up pronunciation for 'kokoro'
            self.pipeline.g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
            
            print("✅ Kokoro-82M initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize Kokoro-82M: {e}")
            print("Make sure you've downloaded the model: hf download hexgrad/Kokoro-82M")
            print("And installed kokoro: pip install kokoro>=0.9.4")
            raise
    
    def filter_transcript_for_tts(self, text):
        """Remove stage directions and visual cues in square brackets for TTS narration"""
        import re
        
        # Remove content in square brackets (stage directions/visual cues)
        cleaned_text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
        
        # Clean up extra whitespace and newlines
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Normalize paragraph breaks
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize spaces
        
        # Remove quotes around spoken text for cleaner TTS
        cleaned_text = cleaned_text.replace('"', '')
        
        return cleaned_text.strip()
    
    def generate_audio(self, text, voice='af_heart', speed=1.0, output_file=None):
        """
        Generate audio from text using Kokoro-82M
        
        Args:
            text: Text to convert to speech
            voice: Voice ID ('af_heart' or 'af_bella')
            speed: Speech speed (0.5-2.0)
            output_file: Output WAV file path
        
        Returns:
            Path to generated audio file
        """
        try:
            print(f"🎤 Generating audio with Kokoro-82M voice '{voice}' at speed {speed}...")
            
            # Text should already be cleaned by narration script conversion
            text = text.strip()
            print(f"📝 Narration script length: {len(text)} characters")
            
            # Use af_heart as the primary voice
            voice = 'af_heart'
            
            # Load voice pack
            pack = self.pipeline.load_voice(voice)
            
            # Generate audio - collect ALL segments, not just first
            audio_segments = []
            for _, ps, _ in self.pipeline(text, voice, speed):
                ref_s = pack[len(ps)-1]
                audio = self.model(ps, ref_s, speed)
                audio_segments.append(audio.numpy())
            
            # Concatenate all audio segments for complete narration
            if audio_segments:
                audio_data = np.concatenate(audio_segments)
            else:
                audio_data = None
            
            if audio_data is None:
                raise Exception("No audio generated")
            
            # Create output filename if not provided
            if output_file is None:
                output_file = f"kokoro82m_{voice}_{int(speed*10)}.wav"
            
            # Save audio as WAV file
            sample_rate = 24000
            wavfile.write(output_file, sample_rate, audio_data)
            
            print(f"✅ Audio generated: {output_file}")
            print(f"📊 Duration: {len(audio_data) / sample_rate:.2f} seconds")
            
            return output_file
            
        except Exception as e:
            print(f"❌ Audio generation failed: {e}")
            return None
    
    def get_available_voices(self):
        """Get list of available voices for Kokoro-82M"""
        voices = {
            'af_heart': '🇺🇸 🚺 Heart ❤️ (Primary Voice)',
        }
        return voices
    
    def test_generation(self, test_text="Hello! This is a test of Kokoro 82M text-to-speech integration for educational videos."):
        """Test audio generation with af_heart"""
        print("🧪 Testing Kokoro-82M generation...")
        
        print(f"Testing voice: af_heart")
        audio_file = self.generate_audio(test_text, voice='af_heart', speed=1.0, 
                                       output_file=f"test_82m_af_heart.wav")
        if audio_file:
            print(f"✅ af_heart: {audio_file}")
            return audio_file
        else:
            print(f"❌ af_heart: Failed")
            return None

def main():
    """Test the Kokoro-82M integration"""
    try:
        # Initialize Kokoro-82M
        kokoro = Kokoro82MGenerator()
        
        # Show available voices
        voices = kokoro.get_available_voices()
        print("🎵 Available voices for Kokoro-82M:")
        for voice_id, voice_name in voices.items():
            print(f"  {voice_id}: {voice_name}")
        print()
        
        # Test generation
        kokoro.test_generation()
        
        print("🎉 Kokoro-82M integration test complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()