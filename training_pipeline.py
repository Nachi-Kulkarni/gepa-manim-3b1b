#!/usr/bin/env python3
"""
Main Training Pipeline - Integrated GEPA optimization for both Code and Transcript generation systems.
"""

import dspy
from dspy import GEPA
import json
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import our systems
from code_generator import (
    CodeGenerator, 
    create_code_gepa_optimizer, 
    evaluate_code_program,
    load_dspy_data
)
from transcript_generator import (
    TranscriptGenerator, 
    create_transcript_gepa_optimizer,
    evaluate_transcript_program,
    load_dspy_transcript_data
)
from code_judge import EnhancedCodeJudge
from transcript_judge import EnhancedTranscriptJudge

class ManimGEPA:
    """
    Main GEPA system that integrates both Code and Transcript generation with optimization.
    """
    
    def __init__(self, api_key: str):
        """Initialize the integrated GEPA system."""
        self.api_key = api_key
        self.setup_language_models()
        self.setup_systems()
        
    def setup_language_models(self):
        """Configure language models for GEPA optimization."""
        logging.info("🔧 Setting up language models...")
        
        # Main LM for generation
        self.main_lm = dspy.LM(
            model="openrouter/google/gemini-2.5-flash",
            api_key=self.api_key,
            temperature=0.7,
            max_tokens=40000,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Reflection LM for optimization
        self.reflection_lm = dspy.LM(
            model="openrouter/google/gemini-2.5-flash",
            api_key=self.api_key,
            temperature=0.6,
            max_tokens=32000,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Configure DSPy
        dspy.configure(lm=self.main_lm)
        logging.info("✅ Language models configured: Gemini 2.5 Flash")
        
    def setup_systems(self):
        """Initialize all component systems."""
        logging.info("🔧 Setting up systems...")
        
        # Base generators
        self.code_generator = CodeGenerator()
        self.transcript_generator = TranscriptGenerator()
        logging.info("✅ Base generators initialized")
        
        # Judges for evaluation
        self.code_judge = EnhancedCodeJudge()
        self.transcript_judge = EnhancedTranscriptJudge()
        logging.info("✅ Judge systems initialized")
        
        # Load data
        self.load_all_data()
        
    def load_all_data(self):
        """Load training and validation data for both systems."""
        try:
            logging.info("📊 Loading training and validation data...")
            
            # Load data for code generation
            self.code_train_data = load_dspy_data('train', 'code')
            self.code_val_data = load_dspy_data('val', 'code')
            
            # Load data for transcript generation
            self.transcript_train_data = load_dspy_transcript_data('train', 'transcript')
            self.transcript_val_data = load_dspy_transcript_data('val', 'transcript')
            
            logging.info(f"✅ Data loaded:")
            logging.info(f"   Code: {len(self.code_train_data)} train, {len(self.code_val_data)} val")
            logging.info(f"   Transcript: {len(self.transcript_train_data)} train, {len(self.transcript_val_data)} val")
            
        except Exception as e:
            logging.error(f"❌ Error loading data: {e}")
            raise
            
    def create_code_optimizer(self):
        """Create GEPA optimizer for code generation."""
        optimizer = GEPA(
            metric=self.code_quality_metric,
            max_metric_calls=50,  # Direct limit for faster testing
            num_threads=2,  # Reduced threads
            track_stats=True,
            reflection_minibatch_size=1,  # Reduced batch size
            reflection_lm=self.reflection_lm
        )
        return optimizer
        
    def create_transcript_optimizer(self):
        """Create GEPA optimizer for transcript generation."""
        optimizer = GEPA(
            metric=self.transcript_quality_metric,
            max_metric_calls=50,  # Direct limit for faster testing
            num_threads=2,  # Reduced threads
            track_stats=True,
            reflection_minibatch_size=1,  # Reduced batch size
            reflection_lm=self.reflection_lm
        )
        return optimizer
        
    def code_quality_metric(self, example, prediction, trace=None, pred_name=None, pred_trace=None):
        """Evaluation metric for code generation."""
        try:
            reference_code = example.get('target_code', '')
            generated_code = prediction.generated_code if hasattr(prediction, 'generated_code') else ''
            
            if not generated_code or not reference_code:
                return 0.0
            
            # Use code judge for evaluation
            evaluation = self.code_judge(
                video_title=example.get('video_title', ''),
                generated_code=generated_code,
                reference_code=reference_code,
                transcript_excerpt=example.get('transcript_excerpt', '')
            )
            
            return evaluation.overall_quality
            
        except Exception as e:
            print(f"Error in code quality metric: {e}")
            return 0.0
            
    def transcript_quality_metric(self, example, prediction, trace=None, pred_name=None, pred_trace=None):
        """Evaluation metric for transcript generation."""
        try:
            reference_transcript = example.get('target_transcript', '')
            generated_transcript = prediction.generated_transcript if hasattr(prediction, 'generated_transcript') else ''
            
            if not generated_transcript or not reference_transcript:
                return 0.0
            
            # Use transcript judge for evaluation
            evaluation = self.transcript_judge(
                video_title=example.get('video_title', ''),
                generated_transcript=generated_transcript,
                reference_transcript=reference_transcript,
                code_excerpt=example.get('code_excerpt', '')
            )
            
            return evaluation.overall_quality
            
        except Exception as e:
            print(f"Error in transcript quality metric: {e}")
            return 0.0
            
    def evaluate_system(self, system_name: str, program, test_data):
        """Evaluate a system on test data."""
        print(f"\n🧪 Evaluating {system_name}...")
        
        if system_name == "Code Generator":
            results = evaluate_code_program(program, test_data)
        elif system_name == "Transcript Generator":
            results = evaluate_transcript_program(program, test_data)
        else:
            raise ValueError(f"Unknown system: {system_name}")
            
        print(f"✅ {system_name} evaluation completed")
        return results
        
    def train_code_gepa(self, max_iterations: int = 10):
        """Train the Code GEPA system."""
        print(f"\n🚀 Training Code GEPA System")
        print("=" * 50)
        
        optimizer = self.create_code_optimizer()
        
        # Evaluate baseline
        baseline_score = self.evaluate_system("Code Generator", self.code_generator, self.code_val_data)
        print(f"📊 Baseline score: {baseline_score}")
        
        # Run optimization
        print(f"🔄 Starting GEPA optimization...")
        optimized_program = optimizer.compile(
            self.code_generator,
            trainset=self.code_train_data[:20],  # Use subset for faster training
            valset=self.code_val_data[:10]
        )
        
        # Evaluate optimized system
        optimized_score = self.evaluate_system("Code Generator", optimized_program, self.code_val_data)
        print(f"📊 Optimized score: {optimized_score}")
        # Extract scores for comparison
        baseline_score_value = baseline_score.score if hasattr(baseline_score, 'score') else baseline_score
        optimized_score_value = optimized_score.score if hasattr(optimized_score, 'score') else optimized_score
        print(f"📈 Improvement: {optimized_score_value - baseline_score_value:.3f}")
        
        return optimized_program
        
    def train_transcript_gepa(self, max_iterations: int = 10):
        """Train the Transcript GEPA system."""
        print(f"\n🎙️ Training Transcript GEPA System")
        print("=" * 50)
        
        optimizer = self.create_transcript_optimizer()
        
        # Evaluate baseline
        baseline_score = self.evaluate_system("Transcript Generator", self.transcript_generator, self.transcript_val_data)
        print(f"📊 Baseline score: {baseline_score}")
        
        # Run optimization
        print(f"🔄 Starting GEPA optimization...")
        optimized_program = optimizer.compile(
            self.transcript_generator,
            trainset=self.transcript_train_data[:20],  # Use subset for faster training
            valset=self.transcript_val_data[:10]
        )
        
        # Evaluate optimized system
        optimized_score = self.evaluate_system("Transcript Generator", optimized_program, self.transcript_val_data)
        print(f"📊 Optimized score: {optimized_score}")
        # Extract scores for comparison
        baseline_score_value = baseline_score.score if hasattr(baseline_score, 'score') else baseline_score
        optimized_score_value = optimized_score.score if hasattr(optimized_score, 'score') else optimized_score
        print(f"📈 Improvement: {optimized_score_value - baseline_score_value:.3f}")
        
        return optimized_program
        
    def run_complete_training(self, code_iterations: int = 10, transcript_iterations: int = 10):
        """Run complete training pipeline for both systems."""
        print("🎯 Complete GEPA Training Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        # Train Code GEPA
        optimized_code_generator = self.train_code_gepa(code_iterations)
        
        # Train Transcript GEPA
        optimized_transcript_generator = self.train_transcript_gepa(transcript_iterations)
        
        # Save results
        self.save_training_results(optimized_code_generator, optimized_transcript_generator)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Complete training finished in {total_time:.1f} seconds")
        
        return {
            'code_generator': optimized_code_generator,
            'transcript_generator': optimized_transcript_generator,
            'training_time': total_time
        }
        
    def save_training_results(self, code_program, transcript_program):
        """Save training results and models."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path('training_results')
        results_dir.mkdir(exist_ok=True)
        
        results = {
            'timestamp': timestamp,
            'training_config': {
                'model': 'openrouter/google/gemini-2.5-flash',
                'code_iterations': 10,
                'transcript_iterations': 10
            },
            'systems_trained': True
        }
        
        # Save results summary
        with open(results_dir / f'training_summary_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Training results saved to {results_dir}")
        
    def demo_systems(self, code_program=None, transcript_program=None):
        """Demonstrate the trained systems."""
        print(f"\n🎭 System Demonstration")
        print("=" * 40)
        
        # Use optimized programs if available, otherwise use base programs
        code_gen = code_program or self.code_generator
        transcript_gen = transcript_program or self.transcript_generator
        
        # Demo with first example from each dataset
        if self.code_train_data:
            code_example = self.code_train_data[0]
            print(f"\n📝 Code Generation Demo:")
            print(f"   Video: {code_example.video_title[:50]}...")
            
            code_prediction = code_gen(
                video_title=code_example.video_title,
                transcript_excerpt=code_example.transcript_excerpt
            )
            
            if hasattr(code_prediction, 'generated_code'):
                print(f"   ✅ Generated {len(code_prediction.generated_code)} chars of code")
                print(f"   📝 Code preview: {code_prediction.generated_code[:200]}...")
            else:
                print(f"   ❌ No code generated")
        
        if self.transcript_train_data:
            transcript_example = self.transcript_train_data[0]
            print(f"\n🎙️ Transcript Generation Demo:")
            print(f"   Video: {transcript_example.video_title[:50]}...")
            
            transcript_prediction = transcript_gen(
                video_title=transcript_example.video_title,
                code_excerpt=transcript_example.code_excerpt[:1000]
            )
            
            if hasattr(transcript_prediction, 'generated_transcript'):
                print(f"   ✅ Generated {len(transcript_prediction.generated_transcript)} chars of transcript")
                print(f"   📝 Transcript preview: {transcript_prediction.generated_transcript[:200]}...")
            else:
                print(f"   ❌ No transcript generated")

def setup_logging():
    """Setup logging for the training pipeline."""
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'training_pipeline_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return log_file

def main():
    """Main training pipeline."""
    # Setup logging
    log_file = setup_logging()
    logging.info("🎯 Manim GEPA Training Pipeline")
    logging.info("=" * 60)
    logging.info(f"Log file: {log_file}")
    
    try:
        # Check API key
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        logging.info("✅ API key found")
        
        # Initialize system
        manim_gepa = ManimGEPA(api_key)
        logging.info("✅ ManimGEPA system initialized")
        
        # Run training
        logging.info("🚀 Starting complete training pipeline...")
        results = manim_gepa.run_complete_training(
            code_iterations=5,  # Reduced for faster demo
            transcript_iterations=5
        )
        
        # Demonstrate results
        logging.info("🎭 Demonstrating optimized systems...")
        manim_gepa.demo_systems(
            results['code_generator'],
            results['transcript_generator']
        )
        
        logging.info("🎉 Training pipeline completed successfully!")
        logging.info("💡 Check training_results/ directory for saved models and results")
        logging.info(f"📝 Full log saved to: {log_file}")
        
    except Exception as e:
        logging.error(f"❌ Error in training pipeline: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()