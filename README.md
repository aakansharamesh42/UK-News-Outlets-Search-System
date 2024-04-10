# TTDS Project: UK News Outlet Search Engine

This project presents a sophisticated search engine tailored specifically for enhancing the accessibility and credibility of news within the UK landscape. By integrating advanced technologies such as TF-IDF scoring, Boolean searches, and sentiment analysis, the system ensures users receive the most relevant and reliable news content. With a focus on quality over quantity, it serves as an invaluable tool for a wide array of users seeking to navigate the extensive array of news with ease and efficiency.

## Features

- Automated aggregation of over 706k news documents from trusted outlets, plus daily live indexing of approximately 1.3k new entries.
- Concise summarization and sentiment assessment to enhance understanding and engagement.
- Advanced search capabilities, including TF-IDF scoring and Boolean searches, to deliver pertinent results.
- Query enhancement tools, such as suggestions and spell-checking, for improved search accuracy.

## Getting Started

### Prerequisites

- Python 3.12.1
- Docker (for containerization)
- Visual Studio 2022 Community Edition (Windows) or build-essential package (Linux) for C++ build tools.

### Environment Setup

#### Virtual Environment (Python)

- **For pyenv**:
```
cd backend
python -m venv .venv
```


- Activate (Windows): `.venv\Scripts\activate`
- Activate (Linux): `source .venv/bin/activate`

- **For conda**:
```
cd backend
conda create -n ttds-proj python=3.12.1
conda activate ttds-proj
```

#### Requirements Installation

- **Windows Requirements**: Install Visual Studio 2022 Community Edition with "Desktop development with C++" and optional MSVC and Windows 11 SDK features.
- **Linux Requirements**:
sudo apt update
sudo apt install build-essential

- **General Requirements** (Install after satisfying OS-specific requirements):
`pip install -r requirements.txt`


### Docker Build and Run

- Build: `docker build -t fastapi:latest .`
- Run: `docker run -d -p 8080:8080 fastapi`

### Deployment

- To deploy, create and push a commit to the "deploy" branch.

## Folder Structure

- **.github/workflows**: CI/CD (build.yml)
- **backend**: Main application code, including routers and utilities.
- **frontend**: React application content (public, src, environment configurations).
- **temp**: Temporary files for local processing and crash recovery (not for production).
