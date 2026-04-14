# Report Summary

## Scope

This project presents a broad segmentation workflow for the Oxford-IIIT Pet dataset, including data preparation, model training, and model comparison steps.

## Workflow

1. Load Oxford-IIIT Pet images and trimap masks.
2. Standardize images and masks to 224x224 with a custom `PetDataset`.
3. Use the official split files for train and test separation.
4. Run exploratory analysis for pixel balance and trimap statistics.
5. Train and compare FCN and U-Net models.
6. Evaluate the effect of data augmentation.
7. Test transfer learning with a pretrained encoder.
8. Add comparisons with Mask R-CNN and YOLOv8 for broader context.

## Main Takeaways

- U-Net based approaches provide better boundary quality than FCN.
- Augmentation improves robustness on difficult examples.
- Transfer learning appears to provide the strongest results in the notebook narrative.
- Detection and segmentation models serve different practical goals.

## Publishing Notes

- Machine-specific paths were removed.
- Notebook outputs were cleared.
- The code was also exported to a `.py` file.
