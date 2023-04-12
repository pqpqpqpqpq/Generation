<h1 align="center"> Design of Microvascular Trees </h1>



<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> : Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project"> ➤ About The Project</a></li>
    <li><a href="#prerequisites"> ➤ Prerequisites</a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#Code-Description"> ➤ Code Description</a></li>
    <!--<li><a href="#experiments">Experiments</a></li>-->
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ABOUT THE PROJECT -->
<h2 id="about-the-project"> : About The Project</h2>

<p align="justify"> 
  This project propose a novel framework to construct heterogeneous microvascular trees. 
  The framework integrates two main algorithm: conditional deep convolutional generative adversarial network 
  (GAN) and a local fractal dimension-oriented constrained constructive optimization
  (LFDO-CCO) strategy.
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PREREQUISITES -->
<h2 id="prerequisites"> : Prerequisites</h2>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python_3.7.12-1f425f.svg)](https://www.python.org/) <br>
[![made-with-MATLAB](https://img.shields.io/badge/Made%20with-MATLAB_R2020a-1f425f.svg)](https://www.mathworks.com/) <br>

<!--This project is written in Python programming language. <br>-->
The following open source packages are used in this project:
* Numpy 1.21.6
* Pandas 0.25.3
* Matplotlib 3.5.1
* TensorFlow-gpu 1.8.0
* Scipy 1.7.3 
* H5py 2.10.0
* Mpmath 1.2.1
* Sympy 1.7.1

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- FOLDER STRUCTURE -->
<h2 id="folder-structure"> : Folder Structure</h2>

    Generation
    .
    │
    ├── Data
    │   ├── Source Data
    │   │   ├── Men_389.dat
    │   │   ├── Men_389.opt
    │   │   ├── Men_546.dat  
    │   │   ├── Men_546.opt
    │   │   ├── Men_913.dat
    │   │   ├── Men_913.opt
    │
    ├── Code
    │   ├── A_1Data2txt.py
    │   ├── A_2DataExtract.py
    │   ├── A_3Support.py
    │   ├── A_4SolveK.m
    │   ├── A_5DataNormalize.py
    │   ├── A_6DataEnhance.py
    │   ├── B_1DSNGAN.py
    │   ├── B_2ChooseBestEpoch.py
    │   ├── C_GetEntrance.py
    │   ├── C_LVDO_CCO.py
    │   ├── C_Func.py
    │   ├── D_1TreeValCV.m
    │   ├── D_2TreeValCV.py
    │   ├── D_2TreeValFDVD.m
    │   ├── D_3OriTreetoExcel.py
    │   ├── D_4OriPrintTree.py
    │   ├── D_5OriValFDVD.m
    │   ├── E_1PrintCVPerGeneration.py
    │   ├── E_2OriPrintCVPerGeneration.py

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- Code-Description -->
<h2 id="Code-Description"> : Code Description</h2>

<p align="justify">
  <ol>
    <li><b>The code is divided into five parts: From A to E</b> </li> 
    <li><b>The code executes in the order of the following table</b></li>
   
  </ol>
  The following table describes the order and description of the code

| Code                                                          | Description                                                                           |
|---------------------------------------------------------------|---------------------------------------------------------------------------------------|
| **A**                                                         | **Data preprocessing**                                                                |
| 📒 **A_1Data2txt.py**                                         | Turn the original data .opt and .dat files into .txt files for subsequent processing. |
| 📒 **A_2DataExtract.py**                                      | Get single bifurcation data.                                                          |
| 📒 **A_3Support.py**                                          | Get the division threshold of the conditions.                                         |
| 📒 **A_4SolveK.m**                                            | Calculate k and λ of each bifurcation.                                                |
| 📒 **A_5DataNormalize.py**                                    | Data normalization.                                                                   |
| 📒 **A_6DataEnhance.py**                                      | Data Enhancement.                                                                     |
| **B**                                                         | **Generate vascular bifurcations**                                                    |
| 📒 **B_1DSNGAN.py**                                           | Model training.                                                                       |
| 📒 **B_2ChooseBestEpoch.py**                                  | Model selecting.                                                                      |
| **C**                                                         | **Generate vascular trees**                                                           |
| 📒 **C_GetEntrance.py**                                       | Get the root bifurcation.                                                             |
| 📒 **C_LVDO_CCO.py**                                          | Generate microvascular trees by assembling the generated bifurcations hierarchically. |
| 📒 **C_Func.py**                                              | The relevant functions in this section.                                               |
| **D**                                                         | **Validation of the generated trees**                                                 |
| 📒 **D_1TreeValCV.m** <br />📒 **D_2TreeValCV.py**            | Calculates the coefficient of variation of the generated trees.                       |
| 📒 **D_2TreeValFDVD.m**                                       | Calculates the fractal dimension and vessel density of the generated trees.           || 📒 **D_3OriTreetoExcel.py** <br />📒 **D_4OriPrintTree.py**  | Calculates the coefficient of variation of the original trees.                       |
| 📒 **D_3OriTreetoExcel.py** <br />📒 **D_4OriPrintTree.py**   | get the data of the real trees per generation and visualize the real trees.           |
| 📒 **D_5OriValFDVD.m**                                        | Calculates the fractal dimension and vessel density of the real trees.                |
| **E**                                                         | **Draw the line graphs of coefficient of variation per generation**                   |
| 📒 **E_1PrintCVPerGeneration.py**                             | Draw the line graphs of coefficient of variation per generation of generated trees.   |
| 📒 **E_2OriPrintCVPerGeneration.py**                          | Draw the line graphs of coefficient of variation per generation of real trees.        |
