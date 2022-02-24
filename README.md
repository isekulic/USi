# USi
This repository contains the code and data for the User Simulator (USi), a model for answering clarifying questions in mixed-initiative conversational search, described in the paper "[Evaluating Mixed-initiative Conversational Search Systems via User Simulation](https://dl.acm.org/doi/abs/10.1145/3488560.3498440)" and presented at WSDM 2022.

### About USi
![Screenshot 2022-02-24 at 19 09 21](https://user-images.githubusercontent.com/9115027/155590977-221fd0d4-1f91-4bca-8dc3-2b5dccedbc09.png)

USi is trained on [Qulac](https://github.com/aliannejadi/qulac/) and [ClariQ](https://github.com/aliannejadi/ClariQ) to answer clarifying questions in line with information need. 
You can use USi for help in evaluation of any models for generating clarifying question on any dataset that has a information need (facet, topic) description (which are a lot of TREC-like collections).

### How to run
Download the pre-trained model [here](drive.google.com) and run predictions:
```
python run.py --test_mode 1 \\
              --test_ckp checkpoints/model_8.ckpt \\
              --temperature 0.7 \\
              --top_k 0 \\
              --top_p 0.9 \\
              --min_output_len 1
```
You can find all the controllable parameters in `argparse` in `run.py`.

### Newly acquired multi-turn dataset

Consistent multi-turn interactions proved difficult for USi. To foster further research on answering clarifying questions in multi-turn interactions, we release a novel multi-turn dataset aimed at constructing conversations with hpyothetical cases, where the clarifying question is repeated, off-topic, or simply ignores the context. 
![Screenshot 2022-02-24 at 19 36 34](https://user-images.githubusercontent.com/9115027/155594885-e1c0d041-b4af-48cc-9dff-72c7b27cabdf.png)

Download the dataset, consisting of 1000 conversations up to the depth of 3, from [here](drive.google.com).

If you found this code or data useful, please cite:
```
@inproceedings{10.1145/3488560.3498440,
    author = {Sekuli\'{c}, Ivan and Aliannejadi, Mohammad and Crestani, Fabio},
    title = {Evaluating Mixed-Initiative Conversational Search Systems via User Simulation},
    year = {2022},
    booktitle = {Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
    pages = {888–896},
    series = {WSDM '22}
}
```

Updates to the repository coming shortly.
