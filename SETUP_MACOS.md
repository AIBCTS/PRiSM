# PRiSM macOS Setup Guide

This guide will walk you through setting up your development environment for the PRiSM project on macOS.

## Prerequisites

- macOS 10.15 Catalina or later
- Internet connection

## Setup Steps

1. **Install Python**
   - Download Python 3.11 from [python.org](https://www.python.org/downloads/)
   - Follow the installation instructions

2. **Install Visual Studio Code**
   - Download VS Code from [code.visualstudio.com](https://code.visualstudio.com/)
   - Move the app to your Applications folder

3. **Open Terminal**

4. **Navigate to Project Directory**
   ```
   cd path/to/your/project
   ```

5. **Create Virtual Environment**
   ```
   make create_environment
   ```
   This will create a virtual environment named `venv_prism` and install the requirements.

6. **Activate Virtual Environment**
   ```
   source venv_prism/bin/activate
   ```

7. **Install ipykernel**
   ```
   pip install ipykernel
   ```

8. **Register Kernel with Jupyter**
   ```
   python -m ipykernel install --user --name=venv_prism
   ```

9. **Open VS Code**
   ```
   code .
   ```

10. **Install VS Code Extensions**
    - Python
    - Jupyter

11. **Select Python Interpreter in VS Code**
    - Open Command Palette (Cmd+Shift+P)
    - Select "Python: Select Interpreter"
    - Choose the interpreter from your `venv_prism` environment

12. **Create New Jupyter Notebook**
    - Click on the "New File" button
    - Select "Jupyter Notebook"

13. **Select Kernel in Notebook**
    - Click on the "Select Kernel" button in the top right
    - Choose "venv_prism" from the list

14. **Test Setup**
    - In a code cell, run:
      ```python
      import sys
      print(sys.executable)
      ```
    This should print the path to the Python interpreter in your virtual environment.