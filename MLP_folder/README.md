# MLP for Image Recognition
Our Dataset:
https://huggingface.co/datasets/aharley/rvl_cdip

## MLP_classifier_combined
This file is meant to be the main program. It combines the functionality of the 3 main program files we developed over the course of the project in attempt to explore as many parameters as possible. All of the variables that one is able to modify are the first set of variables within "if __name__ == '__main__': ". When running this combined program file you can set variables to enable batch processing for higher resolution runs, or enable the grid search variable to test multiple sets of parameters at the same time. The grid search functionality is able to utilize multiprocessing as well, whereas the other ways can not. On top of the provided variables at the top of "if __name__ == '__main__':" you can also manually change the parameter variables in line 191 where we generate clf = MLPClassifier. Adjusting these settings directly in the classifier declaration offers many more parameters to adjust.

## MLP_single_run
This file is meant to be used for a simple, straightforward single run training of the MLP classifier. Set your dataset, image resolution, and MLP classifier parameters and run the code. The trained model will be saved via joblib dump

## MLP_high_res_batching
This file is similar to MLP_single_run except it is designed to utilize higher resolution (>128x128 pixels) image sizes in order to ensure RAM capacity is not an issue.

## MLP_grid_search
This file is similiar to MLP_single_run except it is designed to be able to explore multiple parameter configurations concurrently. Set your variables and parameter_space array to your desired configuration and multi-process away.
