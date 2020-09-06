## AdaptivFloat: A Floating-Point Based Data Type for Resilient Deep Learning Inference

AdaptivFloat is a floating-point inspired number representation format for deep learning that dynamically maximizes and optimally clips its available dynamic range, at a layer granularity, in order to create faithful encoding of neural network parameters.

<img src="images/adf_github.png" width="820" height="250">

AdaptivFloat consistently produces higher inference accuracies compared to block floating-point, uniform, IEEE-like float or posit encodings at low precision (&le; 8-bit) across a diverse set of state-of-the-art neural network topologies.

The table below shows the impact of weight bit compression on the BLEU score (WMT'17 En-to-De) of the Transformer model post-training quantization / post-quantization aware retraining. More results on the paper (see reference below):

| # Bits |                     Float                    |     BFP     |   Uniform   |    Posit    | AdaptivFloat |
|-------|:--------------------------------------------:|:-----------:|:-----------:|:-----------:|:------------:|
| 16    |                  27.4 / 27.4                 | 27.4 / 27.4 | 27.4 / 27.4 | 27.4 / 27.5 |  27.4 / 27.6 |
| 8     |                  27.2 / 27.5                 | 26.3 / 27.3 | 27.3 / 27.4 | 27.3 / 27.5 |  27.3 / 27.7 |
| 7     |                  27.1 / 27.5                 | 16.9 / 26.8 | 26.0 / 27.2 | 27.3 / 27.4 |  27.3 / 27.7 |
| 6     |                  26.5 / 27.1                 |  0.16 / 8.4 | 0.9  / 23.5 | 26.7 / 27.2 |  27.2 / 27.6 |
| 5     |                  24.2 / 25.6                 |  0.0 / 0.0  |  0.0 / 0.0  | 25.8 / 26.6 |  26.4 / 27.3 |
| 4     |                   0.0 / 0.0                  |  0.0 / 0.0  |  0.0 / 0.0  |  0.0 / 0.0  |  16.3 / 25.5 |

## Algorithm

The base algorithm can be found in the script [`adaptivfloat.py`](https://github.com/ttambe/AdaptivFloat/blob/master/adaptivfloat.py).
It can be easily invoked in any ML framework (PyTorch, TensorFlow, etc.) with a Python backend. The example below shows how one can quantize DNN layers' weights to AdaptivFloat format in PyTorch. The user just needs to be specify the desired quantization bit width `f_bits` of the tensor and its number of floating-point exponent bits `f_exp`.

```python
import torch
from adaptivfloat import quantize_adaptivfloat

for parameter in self.model.named_parameters():
    params_np = parameter.cpu().data.numpy()
    params_adaptivfloat = quantize_adaptivfloat(params_np, self.f_bits, self.f_exp, bias = None)
    parameter.data = torch.from_numpy(params_adaptivfloat).float().cuda()
```

## Citation

If you find this resource useful, please consider citing the following paper:

```
@article{Tambe2019AdaptivFloatAF,
  title={AdaptivFloat: A Floating-point based Data Type for Resilient Deep Learning Inference},
  author={Thierry Tambe and En-Yu Yang and Zishen Wan and Y. Deng and V. Reddi and Alexander M. Rush and D. Brooks and Gu-Yeon Wei},
  journal={ArXiv},
  year={2019},
  volume={abs/1909.13271}
}
```
