# Counterfactual Dense Contrastive Learning

This repository provides code and resources for training and evaluating counterfactual-based contrastive learning models on chest X-ray datasets.

---

## **Setup**

### **1. Install Dependencies**

You need to install **conda**, **mamba**, or **micromamba** before proceeding.

Then, download the required datasets:

* [PadChest dataset](https://bimcv.cipf.es/bimcv-projects/padchest/)
* [CheXMask dataset](https://physionet.org/content/chexmask-cxr-segmentation-data/1.0.0/)

> **Note:** For access to the manually annotated subset of **PadChest**, contact the [BioMedia lab](https://biomedia.doc.ic.ac.uk).

---

### **2. Clone the Repository and Create the Environment**

```bash
git clone https://github.com/lafarguem/cf-dense-contrastive-learning
cd cf-dense-contrastive-learning
conda env create -f environment.yaml
conda activate cf-contrastive-seg
```

---

### **3. Configure Dataset Paths**

Modify the file:

```
counterfactuals/data_handling/xray.py
```

Update the dataset paths to point to your downloaded **PadChest** and **CheXMask** datasets.

---

### **4. Generate Counterfactuals**

Run the following commands to start training and generate counterfactuals:

```bash
python -m counterfactuals.main --hps padchest
python -m counterfactuals.generate --hps padchest --resume path/to/hvae.pt
```

---

### **5. Preprocess PadChest Data**

Edit the following files and set the correct paths to the generated folders:

* `preprocessing/manual_padchest.py`
* `preprocessing/padchest.py`

Then, run:

```bash
python -m preprocessing.padchest
python -m preprocessing.manual_padchest
```

---

### **6. Update Dataset Configurations**

Set the dataset paths inside the `.yaml` configuration files located at:

```
configs/data/datasets/
```

Update these files in both the **pretraining** and **transfer** directories to point to the correct folders and files.

---

### **7. Run Pretraining and Transfer Learning**

Once your paths are configured and data is preprocessed, you can run any pretraining experiment available in:

```
pretraining/configs/experiments/
```

For example:

```bash
python -m pretraining.main +experiment=dvd_cl
python -m transfer.main +experiment=default train.pretrained_model=path/to/weights/best.pt
```
---

## **Adding New Components**

To add a new module or component to the pipeline:

1. **Create the implementation** by following the interface of existing components.
2. **Add a YAML configuration file** in the appropriate folder.
3. Set the `target` attribute in your YAML file to point to your new module and define the necessary arguments.

Example:

```yaml
_target_: my_project.models.MyNewModel
param_1: value1
param_2: value2
```

Hydra will automatically instantiate your component based on this configuration.

---

## **References**

* [Hydra Documentation](https://hydra.cc/docs/advanced/instantiate_objects/overview/)
* [PadChest Dataset](https://bimcv.cipf.es/bimcv-projects/padchest/)
* [CheXMask Dataset](https://physionet.org/content/chexmask-cxr-segmentation-data/1.0.0/)
* [CF-seg Dataset](https://arxiv.org/abs/2506.16213)

---

## **License**

This project is licensed under the [MIT License](LICENSE).
