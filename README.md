**README**

To effectively utilize the provided repository, it's essential to clone the repository. Subsequently, 
you can install the required libraries from the "requirements.txt" file to acquire the necessary 
dependencies for this repository. The entirety of this repository's code is written in Python. Nonetheless,
for evaluating the Spectral Angle Mapper (SAM) and Root Mean Square Error (RMSE), Matlab was employed. Similarly,
Matlab was utilized for generating the plotted spectra.

Here's the updated version of the repository.

The main file for the network is the TestFlattenAutoencoder. To ensure proper functioning, 
you need to establish the subsequent folder structure:

**if you use this code please cite this "A blind convolutional deep autoencoder for
spectral unmixing of hyperspectral images over waterbodies"**

**URL**

**https://www.frontiersin.org/articles/10.3389/feart.2023.1229704/full**

**bibtex:**

**@article{AlfaroMeja2023,
  doi = {10.3389/feart.2023.1229704},
  url = {https://doi.org/10.3389/feart.2023.1229704},
  year = {2023},
  month = oct,
  publisher = {Frontiers Media {SA}},
  volume = {11},
  author = {Estefan{\'{\i}}a Alfaro-Mej{\'{\i}}a and Vidya Manian and Joseph D. Ortiz and Roger P. Tokars},
  title = {A blind convolutional deep autoencoder for spectral unmixing of hyperspectral images over waterbodies},
  journal = {Frontiers in Earth Science}
}**

**How do I get set up?**

autoenconder_owt:

--data

--model

--results

--utils

requirements.txt
Readme.md
.gitignore

Inside each folder the following file must be in the content folder:

**data:**

images
ground truth

As an example:

JasperRidge.mat

JasperEnd4.mat

**model:**

all the models can be read here

**results**

It must be the configuration file

*configuration_file.json

TestFlattenAutoencoder.py is the main file where the basic configuration is established.

**To perform the test, you need to modify the following parameters in TestFlattenAutoencoder.py:**

data_path = '../data/samson_1.mat'  # Path to the data; for this case, the experiment uses the samson_1.mat dataset

ground_truth_path = '../data/end3.mat'  # Path to the ground-truth data

results_path = '../results/'  # Directory to save results and models

data_target = 'V'  # Key used for samson dataset; for Jasper, change to "Y"

ground_truth_target = 'M'  # Key for ground-truth in Samson dataset; same for Jasper

ground_truth_abundances_label = 'A'  # Key for GT abundances maps in Samson dataset; same for Jasper

patch_size = 32  # Size of patches to extract from input images

image_size = 95  # Image size for Samson; adjust based on image dimensions; for Jasper, it's 100

image_width = 95  # Image width for Samson; adjust based on image dimensions; for Jasper, it's 100

image_height = 95  # Image height for Samson; adjust based on image dimensions; for Jasper, it's 100

batch_size = 20

rows_number = 10

columns_number = 10

batch_size_depth = 156  # Number of bands for samson is 156; for Jasper, it's 198. Change based on bands number

training_percentage = 70

testing_percentage = 20

validation_percentage = 10

number_images = 8

end_members_number = 4  # Adjust based on the number of endmembers in the dataset

case_option_abundances_maps = 0

number_patches_width = 3

number_patches_height = 3

patch_size_model = 40

number_patches = 250

batch_size_model = 15

''' configuration_file = 'configuration_file_DCNN_SU' '''


configuration_file = 'samson_configuration_file_DCNN_SU'  # This JSON file configures network parameters for the Samson dataset.
The algorithm was executed on a Dell Precision Server 7920 Rack XCTO Base, equipped with an Intel Xeon Gold processor, 
a 4GB Nvidia T1000 graphics card, a 1 TB SATA hard drive, and 64GB of RAM. 
If the computer you are using to run this code does not possess similar resources, 
you can adjust the parameters within the configuration file to match the image size, as shown below:

"patch_size_model": 100,

"batch_size": 100,


**For Jasper, you can use the provided configuration_file_DCNN_SU.json. The configuration is shared in the results folder
Parameters are configured as follows:**

"data_preparation_title": "data_conf_jasper_run_10",  # You can change these filenames to save results in .mat extension

"title_dictionary_end": "end_data_conf_jasper_run_10",  # You can change these filenames to save results in .mat extension

"title_dictionary_ab": "abundances_data_conf_jasper_run_10"  # You can change these filenames to save results in .mat extension


**TestFlattenAutoencoder.py saves the data in a 4D tensor. To convert it to a 3D tensor, use 
"ResultsVisualizationPatches.py". Inside this .py file, configure image_width and image_size.
Then save the 3D tensor in a Matlab file like this:**

data_jasper_run_10 = {
    "endmembers_jasper_10": reconstructed_end_members,
    "abundances_jasper_10": reconstructed_data_abundances
}
savemat(os.path.join("..", "data", 'jasper_run_10.mat'), data_jasper_run_10)

In this case, files are saved in the data folder.

**Once you have the 3D tensor, you can read it in MATLAB to visualize endmembers and abundance maps.
You can also assess the model using the SAD metrics and RMSE.**

**To utilize the provided code in Matlab, follow these steps:**

1. Run samson_abundances_nsae.m to generate abundance maps and compute the RMSE metric. Make sure to include all 
the files provided within the folder.
2. To show the comparison between the endmembers extracted by our NSAE-SU and those obtained for UnDIP, and CNNAE, 
execute end_members_comp_samson.m. Ensure that you have all the necessary files and resources available 
before running the code.

**Finally, the utils folder provides all the .py developed for this publication.**

**Who do I talk to?**

**estefania.alfaro@upr.edu**
