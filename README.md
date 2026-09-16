# DataMatrix

Code and experimental data for the paper:

**DataMatrix Code Recognition Method Based on Coarse Positioning of Images**  
Lingyue Hu, Guanbin Zhong, Zhiwei Chen, and Zhong Chen  
*Electronics*, 2025, 14(12), 2395  
[Paper](https://doi.org/10.3390/electronics14122395)

The method reconstructs DataMatrix codes from degraded images using adaptive grid segmentation, outlier correction, and grayscale prediction. It was evaluated against libdmtx and ZXing on two datasets containing 30 source images.

## Files

- `code/decode.py`: reconstructs the module matrix from a selected image.
- `code/test_image.py`: batch processing, decoding, and runtime/memory measurements.
- `code/canvas.py`: visualizes the sampling grid.
- `image/`: example images.
- `supplement_data/`: supplementary figures and datasets.

## Usage

The scripts use Python, OpenCV, NumPy, pandas, scikit-learn, statsmodels, and Tkinter. Batch processing also requires pylibdmtx, psutil, and openpyxl.

PNG images and ZIP archives use Git LFS. Run `git lfs pull` after cloning.

```bash
python code/decode.py
```

Select a cropped, approximately axis-aligned square DataMatrix image in the file dialog. The script prints the reconstructed matrix.

These are the original research scripts. See the paper for experimental details and limitations.
