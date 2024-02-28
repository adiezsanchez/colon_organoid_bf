# colon_organoid_bf

Analysis of brighfield images of colonoids growing in Matrigel droplets to extract morphological features and object counts.

<h2>Instructions</h2>

1. In order to run these Jupyter notebooks and .py scripts you will need to familiarize yourself with the use of Python virtual environments using Mamba. See instructions [here](https://biapol.github.io/blog/mara_lampert/getting_started_with_mambaforge_and_python/readme.html).

2. Then you will need to create a virtual environment either using the following command or recreate the environment from the .yml file you can find in the envs folder:

   <code>mamba create -n napari-devbio python=3.9 devbio-napari plotly pyqt -c conda-forge</code>

3. Create a data folder in the root directory. Place all of your folders inside the <code>./data/</code> directory.

4. Activate your venv and launch a JupyterLab server by typing this in your command line prompt:

   <code>mamba activate napari-devbio</code>

   <code>jupyter lab</code>

5. Open <code>1_image_analysis.ipynb</code> and run all cells.
