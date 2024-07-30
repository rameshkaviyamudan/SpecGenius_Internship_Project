# Conductive Film Specification Extractor

## Table of Contents
- [Project Overview](#project-overview)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Known Issues and Limitations](#known-issues-and-limitations)
- [Future Improvements](#future-improvements)
- [Contact Information](#contact-information)

## Project Overview
[Provide a brief description of your project, its purpose, and main features]

## System Requirements

- Python 3.8.1
- Node.js (version: [Add version number])
- CUDA 12.4 (optional, for faster extraction)
- GPU (recommended for optimal performance)

## Project Structure
```plaintext
project/
├── app.py
├── run_server.py
├── using_nougat.ipynb
├── finetune.ipynb
├── evaluate.ipynb
├── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   ├── CSVDisplay.js
│   │   └── QueryTable.js
│   ├── package.json
│   └── frontend_dependencies.txt
├── output/
│   └── specifications/
├── uploads/
│   └── output/
└── evaluation_data/
```

- `app.py`: Main backend server file
- `frontend/`: React frontend application
- `output/`: Stores extracted specifications
  - `specifications/`: Contains relevant conductive film specifications
- `uploads/output/`: Stores extracted contents from uploaded papers as MMD files
- `evaluation_data/`: Contains files used for evaluation

## Installation

### Backend Setup

1. Install Python 3.8.1
2. Clone the repository:
git clone [your-repo-url]
cd [your-repo-name]
3. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
4. Install required packages:
pip install -r requirements.txt

### Frontend Setup

1. Navigate to the frontend directory:
cd frontend
2. Install Node.js dependencies:
npm install

### CUDA Setup (Optional)

Install CUDA 12.4 for GPU acceleration. Follow the [official NVIDIA CUDA installation guide](https://developer.nvidia.com/cuda-downloads).

## Configuration

1. Create a `.env` file in the project root directory
2. Add your OpenAI API key:
OPENAI_API_KEY=your_api_key_here

## Running the Application

### Start the Backend Server
`python run_server.py`

### Start the Frontend Application

In a new terminal:

`cd frontend`

`npm start`
