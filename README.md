
TODO everything in VSCODE

Docstrings in python should be beneath the function declaration.
2 spaces after function/methods
1 space after Docstrings
follow pep8


	-> The ".venv" folder contains the Python environment that is specific to the MM project. You
	   should use this environment for the development of the ONLY the MM project as it has the 
	   necessary packages for this project installed. It is good practice to build a specific
	   Python environment for each different project.
	   
	-> The "MM Dev" folder has all of the files that you may need to access for developing the MM 
	   and has been structured so 
	   
	   
The micromanipulator_tools project is structured as a standard Python package, promoting modularity and ease of use. Here's a breakdown of its organization:

The root directory contains the main package folder, 'micromanipulator_tools', along with supporting files like setup.py, requirements.txt, and README.md. Inside the main package folder, you'll find:

init.py: This file makes the folder a Python package. It imports and exposes key classes and functions, allowing users to import directly from micromanipulator_tools.

main.py: Contains high-level functionality or entry points for the package.

utils/: A subpackage for utility modules:

init.py: Makes utils a subpackage and manages imports.
nanocontrol.py: Houses the NanoControl class and related functionality.
turntable.py: Contains the Turntable class and its associated methods.
This structure allows for clean imports like:

from micromanipulator_tools import NanoControl, Turntable
The package uses relative imports within its modules, ensuring internal references remain correct regardless of how the package is installed or used. This organization facilitates easy distribution, installation, and usage of the micromanipulator_tools package in various projects.
