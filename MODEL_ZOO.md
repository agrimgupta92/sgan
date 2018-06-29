# Social GAN Model Zoo

We refer our method as SGAN-kVP-N where kV signifies if the model was trained using variety loss (k = 1 essentially means no variety loss) and P signifies usage of our proposed pooling module. At test time we sample multiple times from the model and chose the best prediction in L2 sense for quantitative evaluation. N refers to the number of time we sample from our model during test time. We report two error metrics Average Displacement Error (ADE) and Final Displacement Error (FDE) for t<sub>pred</sub> = 8 and 12 in meters.

These results are better from what were reported in the paper. You can use print_args to get hyper-parameters used for training. For SGAN-20VP-20 we used 'global' as opposed to 'local' as done in the paper.

**SGAN-20V-20**

| Model | ADE<sub>8</sub>  |  ADE<sub>12</sub> | FDE<sub>8</sub>  | FDE<sub>12</sub>  |
|-----|-----|---    |---    |---   |
| `ETH`| 0.58 |0.71 |1.13 |1.29 |
| `Hotel`| 0.36 |0.48 |0.71 |1.02|
| `Univ`| 0.33 |0.56 |0.70 |1.18 |
| `Zara1`| 0.21 |0.34 |0.42 |0.69|
| `Zara2`| 0.21 |0.31|0.42 |0.64|

**SGAN-20VP-20**

| Model | ADE<sub>8</sub>  |  ADE<sub>12</sub> | FDE<sub>8</sub>  | FDE<sub>12</sub>  |
|-----|-----|---    |---    |---   |
| `ETH`| 0.57 |0.77|1.14 |1.39|
| `Hotel`| 0.38 |0.43|0.73 |0.88|
| `Univ`| 0.42 |0.75|0.79 |1.50|
| `Zara1`| 0.22 |0.34|0.43 |0.68|
| `Zara2`| 0.24 |0.36|0.48 |0.73|
