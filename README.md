# Object Shape Approximation


# How to run
Steps:
- Extract the given .zip file to a location of your choice.
- Go to that location and create a python environment and activate it using:
    ```python3
    python -m venv myvenv #Replace myvenv with a virtual environment name of your choice.

    # If you are on Windows, then run this in command prompt (not powershell) to activate the environment:
    myvenv\Scripts\activate

    # If you are on a Linux or Mac, use this:
    source myvenv/bin/activate    
    ```
- Now, run this command from the virtual environment to install all the dependencies:
    ```python3
    pip install -r requirements.txt
    ```
- Now you can run the code by running this command in the venv:
    ```python3
    streamlit run convex_hull.py
    ```
This should open a browser window, where the project shall load.

- To deactivate the virtual environment, just run ```deactivate``` in the virtual environment.