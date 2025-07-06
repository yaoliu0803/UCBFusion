# UCBFusion
Code for UCBFusion:**"Unsupervised Cross-Modal Biomedical Image  Fusion Framework with Dual-Path Detail  Enhancement and Global Context Awareness"**

## ***1.Requirements***
Setup Main Environment
- Python >= 3.7
- PyTorch >= 1.4.0 + cu102 is recommended
- opencv-python = 4.5.1
- matplotlib

## ***2.Data Preparation***
- The datasets can be downloaded from: [John Innes Centre](http://data.jic.ac.uk/Gfp/) .
- Moreover, you can place it in the folder `./datasets/train_images/`
  
## ***3.Train and Test***
You can see the parameter configuration of the paper in `./opts.py`
- Train, run `./train.py`
- Test, select the file according to your needs, run `./test.py` or `./test_more.py` or `./test_mri_pet.py`

## ***4.Citation***
If you have any questions, please create an issue or email to me [liuyao_0803@163.com](liuyao_0803@163.com) .
