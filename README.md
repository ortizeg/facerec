# FaceRec - Face Recognition Made Simple

FaceRec is a Python library for dealing with face recognition training.

Existing face recognition libraries are huge and convoluted, often making it hard to read, improve. The goal is to provide an easy template to try the latest and greatest rapidly with the least boilerplate modifications.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install facrec locally.

```bash
pip install -e .
```

## Usage

```bash
python scripts/train.py fit --config scripts/config.yaml
```

```mermaid
flowchart LR
    Download --> Detect --> Align --> Train --> Evaluate
```

Workflow should follow download dataset, detect and align faces, run training code, evaluate results on test set (LFW). All of this functionality will be added and exposed as time permits.

## Current Results
| Dataset | Input Size | Backbone | Margin | Output Size | Epochs | LFW |
| --- | --- | --- | --- | --- | --- | --- |
| MS1M | 112 | Resnet18 | Softmax | 512 | 30 | 0.94 |
| DigiFace1M | 112 | Resnet18 | Softmax | 512 | 30 | 0.80 |
| DigiFace1M | 112 | Resnet18 | ArcFace | 512 | 30 | 0.94 |

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
