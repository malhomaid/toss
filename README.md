# Tennis Forehand Analysis Project

A comprehensive toolkit for analyzing and comparing tennis forehand techniques between professional and amateur players using computer vision and biomechanical analysis.

## Project Overview

This project uses pose estimation and motion analysis to:
- Compare professional and amateur tennis forehand techniques
- Analyze key biomechanical metrics (joint angles, rotations, trajectories)
- Generate visualizations of swing differences
- Provide phase-normalized comparisons to account for timing differences
- Create detailed reports with actionable feedback

## Core Files

### Main Analysis Pipeline

- **tennis-forehand-comparision.py**: Main entry point that creates the analysis pipeline
- **tennis_metrics_enhanced.py**: Advanced metrics with phase normalization
- **tennis_metrics_analyzer.py**: Core metrics calculation and analysis
- **tennis_forehand_analysis.py**: Base analysis with pose detection

### Utilities

- **pose-landmarker.py**: Simple tool for pose detection on tennis videos
- **frame_rate_adjuster.py**: Utility for standardizing video frame rates
- **tennis_speed_normalized.py**: Creates speed-normalized comparisons
- **create_slow_motion.py**: Generates smooth slow-motion videos for analysis

### Simple Comparison Tools

- **simple_comparison_with_pose.py**: Side-by-side comparison with pose overlay
- **simple_comparison.py**: Basic side-by-side video comparison

## Getting Started

1. **Setup environment**:
   ```
   pipenv install
   pipenv shell
   ```

2. **Run the main analysis**:
   ```
   python tennis-forehand-comparision.py
   ```

3. **View the results**:
   Output files are saved to the `tennis_analysis_output` directory, including:
   - Comparison videos with pose detection
   - Phase-normalized visualizations
   - Metrics reports and analysis summaries

## Project Structure

- `tennis_analysis_output/`: Contains the latest analysis results and demonstrations
- Python scripts: Analysis tools and utilities

## Requirements

- Python 3.8+
- MediaPipe (for pose detection)
- OpenCV (for video processing)
- NumPy, Matplotlib, and Pandas (for analysis and visualization)

All dependencies are specified in the Pipfile.

## Development Notes

See project documentation for coding standards and guidelines.