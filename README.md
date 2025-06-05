# NRHA Reining Score Predictor

An AI-based system for predicting NRHA (National Reining Horse Association) reining scores from video footage.

## Features

- Total score prediction (0-100)
- Per-maneuver scoring (-1.5 to +1.5)
- Penalty detection and quantification
- Maneuver classification
- Pattern-specific requirement evaluation

## Project Structure

```
pocket-judge
    ├──models
        ├──pose
            ├──data
            ├──runs
            ├──scripts
            README.md
        ├──sequence
            ├──data
            ├──runs
            ├──scripts
            README.md (how to train and run)
        ├──score
            ├──data
            ├──runs
            ├──scripts
            README.md (how to train and run)
    ├──configs
    ├──apps
        ├──(web applications)
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

[Usage instructions will be added as the project develops]

## Development

This project uses:

- Black for code formatting
- isort for import sorting
- flake8 for linting
- pytest for testing

## License

[License information to be added]
