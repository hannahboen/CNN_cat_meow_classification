# Machine Learning Classification of Cat Meows 

### Hannah Boen, Gabe LeBlanc, Bella Tarantino, Ethan Tran

## Overview 

This project leverages the Cat Sound Classification Dataset V2 by Yagya Raj Pandeya (2018) to classify domestic cat emotions based on their vocal activity with the goal of improving human-animal communication for caretakers, vets, and researchers.

## Data and Methodology

### Data Collection & Preprocessing

- **Cat Sound Classification Dataset**: Loaded audio files from source, removing augmented files. 
- **Augmented Data**: Added noise and shifted pitch to increase generalizability with known methods.
- **Cleaned**: Trimmed and padded each audio filefor consistent model input.
- **Pickled**: Converted data for faster future loading.

### Model Development & Training

1. **Basic Convolutional Neural Network**: Basic CNN for prototyping and feature selection.
2. **Baseline VGG**: Initial layers and weights of SOTA model VGG16.
3. **Training Callbacks**: Early stopping, linearly decaying, learning rate.
4. **Hyperparameters**: Dense layer neuron count, dropout rate.

### Final Model Training Parmaters
- Number of epochs: 10
- Batch size: 16
- Initial learning rate: 0.001
- Dropout rate: 0.2

## Results and Conclusion

We achieved a test accuracy of > 96% with a high AUC. 

## Code

The final analysis and model code can be found in the `milestone5.ipynb` notebook.

## Running the Project

1. Clone the repository.
2. Install dependencies.
3. Run `milestone5.ipynb` to reproduce the analysis.

## Authors

- Hannah Boen
- Gabe LeBlanc
- Bella Tarantino
- Ethan Tran

## Acknowledgments

Thanks to the CS109B team for their support and to Yagya Raj Pandeya (2018) for the data.
