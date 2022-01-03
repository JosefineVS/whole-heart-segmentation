# Multi-planar whole heart segmentation of 3D CT images using 2D spatial propagation CNN
Whole heart segmentation from cardiac CT scans is a prerequisite for many clinical applications, but manual delineation is a tedious task and subject to both intra- and inter-observer variation. Automating the segmentation process has thus become an increasingly popular task in the field of image analysis, and is generally solved by either using 3D methods, considering the image volume as a whole, or 2D methods, segmenting each slice independently. In the field of deep learning, there are significant limitations regarding 3D networks, including the need for more training examples and GPU memory. The need for GPU memory is usually solved by down sampling the input images, thus losing important information, which is not a necessary sacrifice when employing 2D networks. It would therefore be relevant to exploit the benefits of 2D networks in a configuration, where spatial information across slices is kept, as when using 3D networks.

The proposed method performs multi-class segmentation of cardiac CT scans utilizing 2D convolutional neural networks with a multi-planar approach. Furthermore, spatial propagation is included in the network structure, to ensure spatial consistency through each image volume. The approach keeps the computational assets of 2D methods while addressing 3D issues regarding spatial context. The pipeline is structured in a two-step approach, in which the first step detects the location of the heart and crops a region of interest, and the second step performs multi-class segmentation of the heart structures. The pipeline demonstrated promising results on the MICCAI 2017 Multi-Modality Whole Heart Segmentation challenge data.

Data is available from http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/

# Citing this library
If you are using this code in a publication, please cite our paper using the following bibtex:
```
@inproceedings{sundgaard2020multi,
  title={Multi-planar whole heart segmentation of 3D CT images using 2D spatial propagation CNN},
  author={Sundgaard, Josefine Vilsbøll and Juhl, Kristine Aavild and Kofoed, Klaus Fuglsang and Paulsen, Rasmus R},
  booktitle={Medical Imaging 2020: Image Processing},
  volume={11313},
  pages={113131Y},
  year={2020},
  organization={International Society for Optics and Photonics}
}
```
The paper can be found at: https://doi.org/10.1117/12.2548015

# Running the scripts
First step: Data augmentation
- Rund augmentation.py

Second step: Region crop (Region folder)
- Train multi-planar U-nets using u_net_aug_XX.py (three scripts, one for each direction)
- Predict with networks using predict_region_XX.py
- Crop regions using region_crop.py, this creates new training and testing samples with cropped versions of the original data

Third step: Multi-class segmentation (Segmentation folder)
- Train multi-planar U-nets using unet_context_XX.py
- Predict on test-dataset using predict_seg.py
- Fusion of the three probability maps using fusion.py

Fourth step: Export as nii-files
- Export using export_all_nii.py
