# Conductive Film Specification Extractor

## Table of Contents
- [Project Overview](#project-overview)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)

## Project Overview

### AI-Powered Specification Extraction System

Developed an AI-driven system to automate the extraction and prediction of conductive film specifications from research papers, enhancing R&D efficiency for Roll-to-Roll (R2R) systems.

1. **Flexible Data Handling**: The system accommodates both full dataset usage and optional data partitioning, allowing users to choose between maximizing training data and setting aside a portion for immediate evaluation.
2. **Automated Workflow**: The process from data preparation to model deployment is highly automated, with seamless integration between the backend processing and frontend updates.
3. **Real-time Status Monitoring**: The implementation includes a robust system for tracking the fine-tuning job status, handling both successful completions and cancellations effectively.
4. **User-Centric Design**: The frontend is designed to guide users through the process, with automatic updates to the query interface and clear indications of when evaluation can be performed.
5. **Error Management**: Comprehensive error handling ensures that users are promptly informed of any issues during the fine-tuning process, maintaining transparency and allowing for quick troubleshooting.
6. **Iterative Model Improvement**: By encouraging immediate evaluation after training, the system promotes an iterative approach to model development and refinement.
7. **Efficient Resource Utilization**: The use of a pre-compiled mega_combined.csv file streamlines the fine-tuning process, avoiding redundant data processing steps.


## System Requirements

- Python 3.8.1
- Node.js (version: v22.4.0)
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
`git clone https://github.com/rameshkaviyamudan/SpecGenius_Internship_Project.git`

4. Create and activate a virtual environment:
`python -m venv venv`
`source venv/bin/activate`  # On Windows use venv\Scripts\activate
5. Install required packages:
`pip install -r requirements.txt`


### Prerequisites

Make sure you have Node.js and npm installed. You can download and install Node.js from [nodejs.org](https://nodejs.org/).

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Install additional frontend dependencies listed in `frontend_dependencies.txt`:
   ```bash
   npm install $(cat frontend_dependencies.txt | xargs)
   ```

4. Run the project:
   ```bash
   npm start
   ```

## Running Tests

To run the tests, use the following command:
```bash
npm test
```

## Building the Project

To build the project, use the following command:
```bash
npm run build
```

## Additional Information

For more details on each dependency and their versions, refer to the `frontend_dependencies.txt` file.

## License

This project is licensed under the MIT License.

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
