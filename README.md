# Deep learning homework for team MM 

**Team name:** MM (nem vagyok benne biztos, hogy ezt adtam meg)

**Team members:** Meglécz Máté, A7RBKU

## Project description

The project is about generating synthetic MRI images using a diffusion model. The model is conditioned on the type of the image (T1 or T2), the target resolution, and the mask of the ventricle and the myocardium. The model is trained on a dataset of real MRI images from Semmelweis University. The goal is to generate realistic synthetic images that can be used for training deep learning models for example in image segmentation tasks.

## Related works

My work relies on the works of Kalapos et al. [1] and Stojanovski et al. [2]. The dataset is described in Kalapos et al. [1], and the deep learning model is based on the work of Stojanovski et al. [2]. The dataset is not publicly available, but it can be requested from the authors of Kalapos et al. Some of the data reading and processing code is based on the code provided by Kalapos et al. 

The base concept of my work comes from Stojanovski et al. [2], where they used a diffusion model to generate synthetic ultrasound images. I use a similar approach to generate synthetic MRI images building on their publicly avaulable github repository. Though I extend the capabilities of the model to not only condition on the mask of the ventricle and the myocardium, but to also condition on the type of the image (T1 or T2) and the target resolution to acquire the most realistic images possible.

The future model is going to be the Semantic Diffusion Model (SDM) [3]. This framework directly feeds the semantic layout and noisy image in to the model to generate high-quality and diverse images. This approach has demonstrated superior performance compared to traditional GAN-based methods, particularly in terms of image quality and diversity.


[1]
A. Kalapos et al., “Automated T1 and T2 mapping segmentation on cardiovascular magnetic resonance imaging using deep learning,” Frontiers in Cardiovascular Medicine, vol. 10, p. 1147581, 2023.

[2]
D. Stojanovski, U. Hermida, P. Lamata, A. Beqiri, and A. Gomez, “Echo from noise: synthetic ultrasound image generation using diffusion models for real image segmentation,” in International Workshop on Advances in Simplifying Medical Ultrasound, 2023, pp. 34–43.

[3]
W. Wang et al., “Semantic image synthesis via diffusion models,” arXiv preprint arXiv:2207.00050, 2022.
<https://github.com/WeilunWang/semantic-diffusion-model>

## Usage

### Docker

To build the docker image, run the following command in the root directory of the repository:
`docker build -t dl-hw-mm .`

Run the docker container with the following command:
`docker run -it --rm dl-hw-mm`

Inside the container, you should clone the repository (I could also copy the files into the container, but I think it is better to clone the repository):
`git clone https://github.com/matemeglecz/dl-hw-mm.git`

### Conda environment

In this case you need cuda already installed on your machine if you want to use the GPU.
I used cuda 11.3.

Alternatively, you can also create the environment from the `requirements.txt` file:
1. `conda create --name dl-hw-mm python=3.8`
2. `conda activate dl-hw-mm`
3. `pip install -r requirements.txt`

### Running the solution

At this point you can run the `data_analysis.ipynb` notebook to analyze the dataset and preprocess the images and masks. The notebook contains the functions to one-hot encode the masks, so they can be used for training the model.

For this I suggest building `Dockerfile_ssh` and ssh into the container to run the notebook from your favorite IDE. 

The dataset should also be mounted into the container, adjust the path in the notebook accordingly.
(The same way the mounting of the code is possible, then cloning is not necessary.)

For mounting use:
`docker run -it --rm -v /path/to/dataset:/path/in/container dl-hw-mm`

## Functions of the files in the repository

- `mapping_utils.py`: Contains utility functions for reading and processing the dataset's dicom files and functions to construct the masks from contours.

- `se_dataset.py`: Contains the dataset class for the synthetic MRI images, does the preprocessing of the images and masks, and returns the images and masks in the correct format for the model. Masks are not one-hot encoded yet, so they can be plotted properly, but they are ready to be one-hot encoded.

- `data_analysis.ipynb`: Contains the data analysis of the dataset, including the distribution of the images and masks, general introduction to the dataset, and examples of the images and masks. Also the function for mask one-hot encoding is in this notebook.