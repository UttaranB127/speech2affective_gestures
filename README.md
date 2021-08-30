# Speech2AffectiveGestures: Synthesizing Co-Speech Gestures with Generative Adversarial Affective Expression Learning

This is the readme to use the official code for the paper [Speech2AffectiveGestures: Synthesizing Co-Speech Gestures with Generative Adversarial Affective Expression Learning](https://arxiv.org/pdf/2108.00262.pdf). Please use the following citation if you find our work useful:

```
@inproceedings{bhattacharya2021speech2affectivegestures,
author = {Bhattacharya, Uttaran and Childs, Elizabeth and Rewkowski, Nicholas and Manocha, Dinesh},
title = {Speech2AffectiveGestures: Synthesizing Co-Speech Gestures with Generative Adversarial Affective Expression Learning},
year = {2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
series = {MM '21}
}
```

## Installation
Our scripts have been tested on Ubuntu 18.04 LTS with
- Python 3.7
- Cuda 10.2
- cudNN 7.6.5
- PyTorch 1.5

1. Clone this repository.

We use $BASE to refer to the base directory for this project (the directory containing `main_v2.py`). Change present working directory to $BASE.

2. [Optional but recommended] Create a conda envrionment for the project and activate it.

```
conda create s2ag-env python=3.7
conda activate s2ag-env
```

3. Install PyTorch via conda.

```
conda install pytorch torchvision torchaudio torchtext cudatoolkit=10.2 -c pytorch
```
Note: You might need to manually uninstall and reinstall `numpy` for `torch` to work.

4. Install the package requirements.

```
pip install -r requirements.txt
```
Note: You might need to manually uninstall and reinstall `matplotlib` and `kiwisolver` for them to work.

## Downloading the datasets
1. The Ted Gestures dataset is available for download [here](https://kaistackr-my.sharepoint.com/:u:/g/personal/zeroyy_kaist_ac_kr/EYAPLf8Hvn9Oq9GMljHDTK4BRab7rl9hAOcnjkriqL8qSg), originally hosted at [https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context).

2. The Trinity Gesture dataset is available for download on submitting an access request [here](https://trinityspeechgesture.scss.tcd.ie/).

## Running the code
Run the `main_v2.py` file with the appropriate command line arguments.
```
python main_v2.py <args list>
```

The full list of arguments is available inside `main_v2.py`.

For any argument not specificed in the command line, the code uses the default value for that argument.

On running `main_v2.py`, the code will train the network and generate sample gestures post-training.

We also provide a pretrained model for download at [this link](https://drive.google.com/file/d/1os20nWp5fLTn2tLLG4Ekc9OnsJlnFjug/view?usp=sharing). If using this model, save it inside the directory `$BASE/models/ted_db` (create the directory if it does not exist). Set the command-line argument `--train-s2ag` to `False` to skip training and use this model directly for evaluation. The generated samples are stored in the automatically created `render` directory.
