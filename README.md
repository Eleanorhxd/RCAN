# RCAN: Recalibrated Cross-modal Alignment Network for Radiology Report Generation with Weakly Supervised Contrastive Learning


## Requirements
- `Python >= 3.6`
- `Pytorch >= 1.7`
- `torchvison`
`conda activate tencent`

## Data

Download IU and MIMIC-CXR datasets, and place them in `data` folder.

- IU dataset from [here](https://iuhealth.org/find-medical-services/x-rays)
- MIMIC-CXR dataset from [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

## Folder Structure
- config : setup training arguments and data path
- data : store IU and MIMIC dataset
- models:  our model
- modules: 
    - the layer define of our model 
    - dataloader
    - loss function
    - metrics
    - tokenizer
    - some utils
- preprocess: data preprocess
- pycocoevalcap: Microsoft COCO Caption Evaluation Tools

## Training & Testing

The source code for training can be found here：

Run `main_train.py` to train a model on the IU X-Ray data and MIMIC-CXR data.

Run `main_test.py` to test a model on the IU X-Ray data and MIMIC-CXR.

To run the command, you only need to specify the config file and the GPU ID and iteration version of the model to be used
