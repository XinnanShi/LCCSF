**Prerequisite:**

- Python 3.8+
- PyTorch 1.12+ and corresponding torchvision

**Clone our repository:**

```bash
git clone https://github.com/XinnanShi/LCCSF.git
```

**Install with pip:**

```bash
cd LCCSF
pip install -e .
```

(If you encounter the File "setup.py" not found error, upgrade your pip with pip install --upgrade pip)

# Training and Evaluation

# Training

## Setting Up Data

The Datasets should list on the folder where main folder"LCCSF" in.

```bash
├── LCCSF
├── DAVIS
└── YouTube
└── MOSE
└── LVOS
```

Links to the datasets:
- DAVIS: https://davischallenge.org/
- YouTubeVOS: https://youtube-vos.org/
- MOSE: https://henghuiding.github.io/MOSE/
- LVOS: https://lingyihongfd.github.io/lvos.github.io/

## Training Command

```
OMP_NUM_THREADS=4 torchrun --master_port 25357 --nproc_per_node=4 LCCSF/train.py exp_id=[some unique id] model=[small/base] data=[base/with-mose/mega]
```

- Change `nproc_per_node` to change the number of GPUs.
- Prepend `CUDA_VISIBLE_DEVICES=...` if you want to use specific GPUs.
- Change `master_port` if you encounter port collision.
- `exp_id` is a unique experiment identifier that does not affect how the training is done.
- Models and visualizations will be saved in `./output/`.
- For pre-training only, specify `main_training.enabled=False`.
- For main training only, specify `pre_training.enabled=False`.
- To load a pre-trained model, e.g., to continue main training from the final model from pre-training, specify `weights=[path to the model]`.

## Example

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 LCCSF/train.py exp_id=test model=base data=base weights=weights/DAVIS.pth pre_training.enabled=False


# Evaluation

## Setting Up Data
For the LVOS validation set, pre-process it by keeping only the first annotations:

```bash
python scripts/data/preprocess_lvos.py ../LVOS/valid/Annotations ../LVOS/valid/Annotations_first_only
```
## Evaluation Command
```
CUDA_VISIBLE_DEVICES=6 python LCCSF/eval_vos.py dataset=[d17-val/d17-test-dev/y18-val/y19-val/...] weights=./output/test3/test3_main_training_last.pth model=base 
```

If you want to use other datasets, please add it on LCCSF/LCCSF/config/evalbase.yaml

You can get our weights at: https://github.com/XinnanShi/LCCSF/releases/tag/v1.0

Some paths and variables in the code may cause issues depending on the system. Please adjust them by yourself

## References

- We develop our code based on the [Putting the Object Back into Video Object Segmentation](https://hkchengrex.github.io/Cutie)
