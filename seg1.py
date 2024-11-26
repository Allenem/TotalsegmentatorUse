import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

if __name__ == "__main__":
    
    input_path, output_path = './img/MRT1_0028_0000.nii', './label/output1'
    
    # option 2: provide input and output as nifti image objects
    input_img = nib.load(input_path)
    output_img = totalsegmentator(input=input_img, output=output_path, task="total_mr", roi_subset=['vertebrae'])
    # nib.save(output_img, output_path)