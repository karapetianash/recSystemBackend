# Movie Recommendation System - Python Part

This part of the project handles data preprocessing, model training, recommendation generation, and evaluation.

## Table of Contents

- [Overview](#overview)
- [Structure](#structure)
- [Setup](#setup)
- [Usage](#usage)
- [License](#license)

## Overview

The Python part of the project reads movie and ratings data, preprocesses it, splits it into training and test sets, trains a collaborative filtering model, generates recommendations, and evaluates the model's performance.

## Structure
    
    python/
    ├── Dockerfile
    ├── requirements.txt
    ├── preprocess.py
    ├── main.py
    ├── database.py
    ├── similarity.py
    ├── recommendations.py
    └── tests/
        ├── test_database.py
        ├── test_integration.py
        ├── test_preprocess.py
        ├── test_recommendations.py
        └── test_similarity.py


## Setup

### Prerequisites

- Python 3.x
- Install dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

Run the preprocessing script to preprocess the data and split it into training and test sets:

```bash
python preprocess.py
```
### Model Training and Recommendation Generation

Run the main script to train the model, generate recommendations, and evaluate the model:

```bash
python main.py
```

### Running Tests
```bash
pytest -s tests
```

## License
This project is licensed under the MIT License.