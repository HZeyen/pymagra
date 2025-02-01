# Pymagra

Software package for treatment of potential field data (magnetics and gravity).
It is mainly thought for treatment of Geometrics gradiometer data (.stn format), but several other data formats are read like Surfer format (Geometrics exportation .dat), gxf or BRGM aeromagnetic files.

**This project is under active development**

Installation:

`pip install pymagra`

Once installed, open an Anaconda Powershell window, cd to the folder containing the data file(s) and type:

`pymagra` followed by `Enter`.

This may be done also from any other command window (cmd under Windows) if the folder containing the executable is placed in the Path. In Windows, this path is C:\User\user_name\anaconda3\Skripts, in Linux ~/anaconda3/bin.

You may also open file pymagra_start.py (in C:\User\user_name\anaconda3\Lib\ site-packages\Pymagra) in a GUI like Spyder and run it from there. This gives the user the possibility to define a working directory at the beginning of pymagra (near line 18, search: “dir0 =” and change the given path), especially useful if you a working for several days with data in the same folder. If using Spyder, it is recommended to run the program within its own console: “Run -> Configuration per file -> Run with custom configuration -> Execute in dedicated console”.

Read the **[manual](./docs/pymagra_Manual.pdf)** for installation, menu description and including a short tutorial.

**API html pages** may be found [here](./docs/build/html). 
