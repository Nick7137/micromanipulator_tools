TODO

create video on setup


talk about main py
talk about utils

add link to github

THE LATEST VERSION IS ON GITHUB



---------------- Installing the Python environment on your personal machine ----------------

This tutorial will help you create a Python virtual environment that is identical to that on 
the lab computer for use on YOUR OWN computer. Copy the Micromanipulator Tools folder to your
machine and navigate to the following directory in VSCode:

	Micromanipulator Tools\1. Software Development\1. Python
	
Read the rest of this section before following this tutorial to use VSCode on your computer:
https://code.visualstudio.com/docs/python/python-tutorial. 

ENSURE you install python version 3.11.7. On the lab computer I created the .venv folder not 
in the "Micromanipulator Tools" folder and placed it in the following directory:

	TODO

I did this to keep the size of "Micromanipulator Tools" small so it can be easily copied to 
a USB. In the tutorial when you get to creating your own virtual environment do the following
instead to create the .venv folder in a different directory.

Open Windows PowerShell and navigate to the folder you want to create your environment in. Then 
enter this command to create an environment called "esaVenv":

>>> py -m venv esaVenv

Then in VSCode type "CTL + SHIFT + P" and pick "Python: Select Interpreter" then click 
"Select Interpreter Path" and find the python.exe file that is in the venv you created.

Now to allow the PowerShell terminal to work in VSCode you will need to open PowerShell with 
admin privaliges from Windows and run the following command:

>>> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

When you get to the install packages section of the tutorial, don't freeze the requirements 
file instead install the requirements.txt that is in the Python folder.
I.e. in the VSCode terminal, run the following command using your newly created .venv environment:

>>> pip install -r requirements.txt

---------------------------------------------------------------------------------------------

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
