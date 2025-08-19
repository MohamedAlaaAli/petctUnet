# import os
# import time
# from moosez import moose

# def main():
#     # Inputs
#     pth = "CTres.nii.gz"
#     output_dir = "organ segmentations"
#     tasks = ['clin_ct_organs', 'clin_ct_digestive']
#     device = "cuda"

#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Time the Moose segmentation
#     start = time.time()
#     moose(pth, tasks, output_dir, device)
#     print(f"✓ Moose completed for {len(tasks)} tasks in {time.time() - start:.2f} seconds")

# if __name__ == "__main__":
#     import multiprocessing
#     multiprocessing.freeze_support()  
#     main()
