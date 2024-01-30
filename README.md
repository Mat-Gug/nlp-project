# NLP Project: Email Analysis :e-mail::mag:

Hello everyone, thank you for being here! 😊

The project aims to develop a library capable of analyzing received emails. In particular, the tasks include:

* Training a classifier to identify SPAM emails.
* Identifying the main topics within the SPAM emails in the dataset.
* Calculating the semantic distance between the obtained topics to deduce their heterogeneity.
* Extracting organizations from the non-SPAM emails.

The provided dataset is `spam_dataset.csv`.

You can access the project by referring to the `Email Analysis Project.ipynb` notebook.<br>
Additionally, the Python library `email_analysis_library` contains all the custom functions utilized throughout the project, which are divided into the modules `preprocessing`, `spam_detection`, `lda`, and `ner`. Finally, `requirements.txt` can be employed to install the necessary dependencies in the project's virtual environment (Python version: 3.9.0). Detailed instructions for setting up the virtual environment are provided below.

## Setting Up a Virtual Environment and Installing Dependencies

Before running the project, it's considered a best practice to create a virtual environment and install the required dependencies. This helps isolate project-specific dependencies from system-wide Python packages. To achieve this, follow these steps:

1. **Create a Virtual Environment:**
- For Windows:
  - To create a virtual environment with the default Python version:
    ```
    python -m venv toxic_venv
    ```
  - To create a virtual environment with a specific Python version, such as Python 3.11, replace `python` with `py` followed by the desired Python version:
    ```
    py -3.11 -m venv toxic_venv
    ```
- For macOS and Linux:
  ```
  python3 -m venv toxic_venv
  ```
2. **Activate the Virtual Environment:**
- For Windows (Command Prompt):
  ```
  toxic_venv\Scripts\activate
  ```
- For Windows (Git Bash):
  ```
  source toxic_venv/Scripts/activate
  ```
- For macOS and Linux:
  ```
  source toxic_venv/bin/activate
  ```
3. **Clone the Repository:**
```
git clone https://github.com/Mat-Gug/toxic_comment_filter.git
```
4. **Navigate to the Project Directory:**
```
cd toxic_comment_filter
```
5. **Install Required Dependencies:**
```
pip install -r requirements.txt
```
6. **Create an IPython Kernel for Jupyter Notebook:**

After activating your virtual environment, run the following command to create an IPython kernel for Jupyter Notebook:
```
python -m ipykernel install --user --name=toxic_venv_kernel
```
If you don't have `ipykernel` installed, you can do it by running the following command:
```
pip install ipykernel
```
7. **Deactivate the Virtual Environment:**

Whenever you're done working on the project, you can deactivate the virtual environment:
```
deactivate
```
By following these steps, you'll have your project set up in an isolated virtual environment with all the required dependencies installed, and you'll be able to use Jupyter Notebook with your project-specific kernel. This is very helpful to ensure that the project runs consistently and without conflicts.
