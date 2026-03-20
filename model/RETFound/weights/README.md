---
tags:
- pytorch
extra_gated_fields:
  First Name: text
  Last Name: text
  Affiliation: text
  Job title:
    type: select
    options:
    - Student
    - Research Graduate
    - AI researcher
    - AI developer/engineer
    - Other
  geo: ip_location
extra_gated_button_content: Submit
---
# Model Card for RETFound_MAE_MEH

<!-- Provide a quick summary of what the model is/does. -->

This modelcard aims to provide a pre-trained vision foundation model [RETFound](https://github.com/rmaphoh/RETFound_MAE), pre-trained with Masked Autoencoder.

This is the official weight for [RETFound Nature paper](https://www.nature.com/articles/s41586-023-06555-x)

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** Yukun Zhou
- **Model type:** Pre-trained model
- **License:** Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [RETFound](https://github.com/rmaphoh/RETFound_MAE)
- **Paper:** [Nature paper](https://www.nature.com/articles/s41586-023-06555-x)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

This repo contains the model weight. After granted the access, please fill the token in the [code](https://github.com/rmaphoh/RETFound_MAE).

The code will automatically download the model and run the training. 



## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->


- **Hardware Type:** 4 * NVIDIA A100 80GB
- **Hours used:** 14 days
- **Cloud Provider:** UCL CS Cluster & Shanghai Jiaotong University Cluster


## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

```
@article{zhou2023foundation,
  title={A foundation model for generalizable disease detection from retinal images},
  author={Zhou, Yukun and Chia, Mark A and Wagner, Siegfried K and Ayhan, Murat S and Williamson, Dominic J and Struyven, Robbert R and Liu, Timing and Xu, Moucheng and Lozano, Mateo G and Woodward-Court, Peter and others},
  journal={Nature},
  volume={622},
  number={7981},
  pages={156--163},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

## Model Card Contact

**ykzhoua@gmail.com** or **yukun.zhou.19@ucl.ac.uk**