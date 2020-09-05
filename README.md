##AdaptivFloat: A Floating-Point Based Data Type for Resilient Deep Learning Inference

AdaptivFloat is a floating-point inspired number representation format for deep learning that dynamically maximizes and optimally clips its available dynamic range, at a layer granularity, in order to create faithful encoding of neural network parameters. AdaptivFloat consistently produces higher inference accuracies compared to block floating-point, uniform, IEEE-like float or posit encodings at very low precision ($\leq$ 8-bit) across a diverse set of state-of-the-art neural network topologies.

Impact of weight bit compression post-training quantization / post-quantization aware retraining

| :----- | :-------------------------------------------: | :----------: | :----------: | :----------: | :---------------: | :--------------------------------------------: | :------------: | :------------: | :------------: | :---------------: | :---------------------------------------------------: | :----------: | :----------: | :----------: | :---------------: |
|        | BLEU Score of Transformer (BLEU @ FP32=27\.4) |              |              |              |                   | Word Error Rate of Seq2Seq (WER @ FP32=13\.34) |                |                |                |                   | Top-1 Accuracy of ResNet-50 (Top-1 Acc. @ FP32=76\.2) |              |              |              |                   |
| \#Bits | Float                                         | BFP          | Uniform      | Posit        | **AdaptivFloat**  | Float                                          | BFP            | Uniform        | Posit          | **AdaptivFloat**  | Float                                                 | BFP          | Uniform      | Posit        | **AdaptivFloat**  |
| 16     | 27\.4 / 27.4                                  | 27\.4 / 27.4 | 27\.4 / 27.4 | 27\.4 / 27.5 | 27\.4 / 27.6      | 13\.40 / 13.07                                 | 13\.30 / 13.14 | 13\.27 / 12.82 | 13\.29 / 13.05 | 13\.27 / 12.93    | 76\.1 / 76.3                                          | 76\.2 / 76.3 | 76\.1 / 76.3 | 76\.1 / 76.3 | 76\.2 / 76.3      |
| 8      | 27\.2 / 27.5                                  | 26\.3 / 27.3 | 27\.3 / 27.4 | 27\.3 / 27.5 | 27\.3 / 27.7      | 14\.06 / 12.74                                 | 13\.23 / 13.01 | 13\.28 / 12.89 | 13\.24 / 12.88 | 13\.11 / 12.59    | 75\.4 / 75.9                                          | 75\.7 / 76.0 | 75\.9 / 76.1 | 75\.4 / 76.0 | 75\.7 / 76.3      |
| 7      | 27\.1 / 27.5                                  | 16\.9 / 26.8 | 26\.0 / 27.2 | 27\.3 / 27.4 | 27\.3 / 27.7      | 13\.95 / 12.84                                 | 13\.54 / 13.27 | 13\.45 / 13.37 | 13\.36 / 12.74 | 13\.19 / 12.80    | 73\.8 / 75.6                                          | 74\.6 / 75.9 | 75\.3 / 75.9 | 74\.1 / 75.8 | 75\.6 / 76.1      |
| 6      | 26\.5 / 27.1                                  | 0\.16 / 8.4  | 0\.9  / 23.5 | 26\.7 / 27.2 | 27\.2 / 27.6      | 15\.53 / 13.48                                 | 14\.72 / 14.74 | 14\.05 / 13.90 | 15\.13 / 13.88 | 13\.19 / 12.93    | 65\.7 / 74.8                                          | 66\.9 / 74.9 | 72\.9 / 75.2 | 68\.8 / 75.0 | 73\.9 / 75.9      |
| 5      | 24\.2 / 25.6                                  | 0\.0 / 0.0   | 0\.0 / 0.0   | 25\.8 / 26.6 | 26\.4 / 27.3      | 20\.86 / 19.63                                 | 21\.28 / 21.18 | 16\.53 / 16.25 | 19\.65 / 19.13 | 15\.027 / 12.78   | 16\.1 / 73.6                                          | 13\.2 / 73.4 | 15\.1 / 74.0 | 33\.0 / 73.9 | 67\.2 / 75.6      |
| 4      | 0\.0 / 0.0                                    | 0\.0 / 0.0   | 0\.0 / 0.0   | 0\.0 / 0.0   | 16\.3 / 25.5      | inf / inf                                      | 76\.05 / 75.65 | 44\.55 / 45.99 | inf / inf      | 19\.82 / 15.84    | 0\.5 / 66.3                                           | 0\.5 / 66.1  | 2\.6 / 67.4  | 0\.7 / 66.7  | 29\.0 / 75.1      |


## Algorithm

The base algorithm can be found in the adaptvfloat.py file

## Citation

If you find this resource useful, please consider citing the following paper:

```
@INPROCEEDINGS{ttambe2020adaptivfloat,
    title={Algorithm-Hardware Co-Design of Adaptive Floating-Point Encodings for Resilient Deep Learning Inference},
    author={Thierry Tambe and En-Yu Yang and Zishen Wan and Y. Deng and V. Reddi and Alexander M. Rush and D. Brooks and Gu-Yeon Wei},
    booktitle={2020 57th ACM/IEEE Design Automation Conference (DAC)}, 
    year={2020},
}
```
