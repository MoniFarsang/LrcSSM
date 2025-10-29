# LrcSSM

This repo contains the code for *Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling* accepted at NeurIPS 2025. 

In this work, we present **LrcSSM** (Liquid-Resistance Liquid-Capacitance State Space Model), a **non-linear** recurrent model that processes long sequences as fast as today’s linear state space layers. By forcing the Jacobian matrix to
be diagonal, the full sequence can be solved in **parallel**. Our model design approach can be generalized to other non-linear recurrent models, demonstrating broader applicability.

Arxiv: [**link**](https://arxiv.org/pdf/2505.21717).

This repo builds upon the [**ELK repo**](https://github.com/lindermanlab/elk).

## Installation Instructions

Following [**ELK repo**](https://github.com/lindermanlab/elk):

We recommend using a virtual environment. **Use python 3.12.1**

Within that virtual environment, first install JAX with
```
pip install --upgrade pip
pip install -U "jax[cuda12]"
```

After installing JAX, pip install the package with
```
pip install --upgrade -e .
```

## Datasets

For ease of use, we added the dataset splits to the repo directly.
They are saved from the script of [https://github.com/tk-rusch/linoss](https://github.com/tk-rusch/linoss/blob/main/data_dir/datasets.py).


## Citation
```bibtex
@misc{farsang2025scalingliquidresistanceliquidcapacitancenetworks,
      title={Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling}, 
      author={Mónika Farsang and Ramin Hasani and Daniela Rus and Radu Grosu},
      year={2025},
      eprint={2505.21717},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.21717}, 
}
```
