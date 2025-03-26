# HLW Dataset 

This package contains the imagery and metadata for the HLW dataset, introduced
in the following publication: 

> Horizon Lines in the Wild (Scott Workman, Menghua Zhai, Nathan
> Jacobs), In arXiv preprint arXiv:1604.02129, 2016.
> [pdf](https://arxiv.org/pdf/1604.02129v1.pdf) 

```
@article{workman2016hlw,
    title={Horizon Lines in the Wild},
    author={Workman, Scott and Zhai, Menghua and Jacobs, Nathan},
    journal={arXiv preprint arXiv:1604.02129},
    year={2016}
}
```

## Included Files 

  * images/ - image data organized by model
  * split/ 
    * train.txt - list of images used for training
    * val.txt - list of images used for validation
    * test_heldout.txt - list of images used for testing (heldout set)
    * test_seen.txt - list of images used for testing (from models in training set)
    * test.txt - union of the two previous lists
  * example.m - demo showing how to parse & visualize the data
  * metadata.csv - csv file containing ground truth
    * format = [image name,
   left horizon point (x),
   left horizon point (y),
   right horizon point (x),
   right horizon point (y)]

## Contact

Scott Workman  
scott@cs.uky.edu  
University of Kentucky  
http://cs.uky.edu/~scott/  

Nathan Jacobs   
jacobs@cs.uky.edu   
University of Kentucky   
http://cs.uky.edu/~jacobs/   
