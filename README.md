# Hetero-Guard

Graph neural networks (GNNs) have achieved remarkable success in many application domains including drug discovery, program analysis, social networks, and cyber security. However, it has been shown that they are not robust against adversarial attacks. In the recent past, many adversarial attacks against homogeneous GNNs and defenses have been proposed. However, most of these attacks and defenses are ineffective on heterogeneous graphs as these algorithms optimize under the assumption that all edge and node types are of the same and further they introduce semantically incorrect edges to perturbed graphs. Here, we first develop, HetePR-BCD, a training time (i.e. poisoning) adversarial attack on heterogeneous graphs that outperforms the start of the art attacks proposed in the literature. Our experimental results on three benchmark heterogeneous graphs show that our attack, with a small perturbation budget of 15%, degrades the performance up to 32% (F1 score) compared to existing ones. It is concerning to mention that existing defenses are not robust against our attack. These defenses primarily modify the GNN’s neural message passing operators assuming that adversarial attacks tend to connect nodes with dissimilar features, but this assumption does not hold in heterogeneous graphs. We construct HeteroGuard, an effective defense against training time attacks including HetePR-BCD on heterogeneous models. HeteroGuard outperforms the existing defenses by 3-8% on F1 score depending on the benchmark dataset.

![heteroguard](https://user-images.githubusercontent.com/9572475/219336616-9cb19f9c-88e7-40cc-98ca-4d5b5f6bc6ee.jpg)

## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email (`udeshk@scorelab.org`). If you encounter any problems when using the code, or want to report a bug, you can open an issue.

## Citation

Please cite our paper if you use your work:

```bibtex
@inproceedings{heteroguard,
  author = {U. Kumarasinghe and M. Nabeel and K. De Zoysa and K. Gunawardana and C. Elvitigala},
  booktitle = {2022 IEEE International Conference on Data Mining Workshops (ICDMW)},
  title = {HeteroGuard: Defending Heterogeneous Graph Neural Networks against Adversarial Attacks},
  year = {2022},
  volume = {},
  issn = {},
  pages = {698-705},
  keywords = {training;drugs;toxicology;social networking (online);perturbation methods;message passing;conferences},
  doi = {10.1109/ICDMW58026.2022.00096},
  url = {https://doi.ieeecomputersociety.org/10.1109/ICDMW58026.2022.00096},
  publisher = {IEEE Computer Society},
  address = {Los Alamitos, CA, USA},
  month = {dec}
}

