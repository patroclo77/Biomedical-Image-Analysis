# Biomedical-Image-Analysis

## Udemy. Set-Up For Windows

Descargar Anaconda, un entorno de gestión de paquetes y desarollo en python.

https://www.anaconda.com/download

Luego abre anaconda prompt y pon los siguientes comandos:

```

conda create -n pytorchenv python=3.10 -y
conda activate pytorchenv
pip config set global.timeout 200


conda install numpy scipy pandas scikit-image scikit-learn matplotlib -c conda-forge

conda install jupyterlab ipykernel ipywidgets -c conda-forge


conda install lightning nibabel pydicom dicom2nifti torchio shapely simpleitk imageio opencv-python tqdm

conda install -c conda-forge tensorboard



conda install notebook -c conda-forge

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install pyqt


```

If IPython is not found:

```
conda remove notebook jupyter jupyterlab ipykernel ipywidgets -y
conda install notebook jupyterlab ipykernel ipywidgets -c conda-forge
```



## Set-Up for Windows


## Libraries for python

python -m pip install imageio matplotlib


## Others

Course from DataCamp

https://campus.datacamp.com/courses/biomedical-image-analysis-in-python
