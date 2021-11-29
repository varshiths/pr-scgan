# seq-cgan  

This is an effort to model sequences and learn translations, JSL from and to Japanese using adversarial methods.  
This work was done as part of an internship at [HRI-JP](https://www.jp.honda-ri.com/en/index.html), in the summer of 2018. Report can be found [here](https://varshiths.github.io/res/HRIJP.pdf).

The current data available is  
- Gloss Annotations of sentences in Japanese  
- Kinect / Video / MoCap data of the sentences signed  

## Setup  

The project uses python3.  
It requires tensorflow and a few other packages listed in the `requirements.txt` file.  

Consider setting up a python environment.  

```bash
pip3 install -r requirements.txt
```

For simplicity, the bash command to run the code has been written in `run.sh` file.  

```bash
bash run.sh
```

For arguments to the command, refer to `main.py`.  

## Project Structure  

The file containing the main run code is contained in `main.py`.  

The models used are placed in the module `models`.  

The configuration of the corresponding models are in the `configs` folder in JSON format.  

The code related to data download and processing is placed in `data` folder.  

## Acknowledgements  

The repository structure is inspired from this [project](https://github.com/MrGemy95/Tensorflow-Project-Template).  

## References  

- [GAN]( https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf )  
- [SeqGAN]( https://arxiv.org/pdf/1609.05473.pdf )  
- [cGAN]( https://arxiv.org/pdf/1411.1784.pdf )  
- [BiGAN]( https://arxiv.org/pdf/1605.09782.pdf )  
- [Age-cGAN]( https://arxiv.org/pdf/1702.01983.pdf )  
- [CANDY]( https://arxiv.org/pdf/1801.02892.pdf )  
- [ConvSeq2Seq]( https://arxiv.org/pdf/1705.03122.pdf )  
- [Tree Conv]( https://arxiv.org/pdf/1409.5718.pdf )  
