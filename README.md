# kaggle-whale-tail
25th solution for the Kagge Competition https://www.kaggle.com/c/humpback-whale-identification

## Env
Look at `requirements.sh`. There are listed all requirements. After creating new conda evv, you can just run `sh requirements.sh`

## Data

Download data to `data` folder. Then apply script `data_cropping.py` for train and test dataset.

## Training

For training first of all change path in main file. There are two variable to change: train_df and train_folder. 
You can also change `batch-size` (default = 16, which need 2x 11 GB).
You can then just run

`
python main_cosine_bb_final.py <folder_name>
`
There would be saved two checkpoints. The last one and with best-acc. As this script learn on entire dataset, I'm using last one (`checkpoint_base`).

## Submission 
Here the code is for only 'without-new-whale'. To create the submission, firstly change folder to data in the file. 

Then 
`eval_bb_se.py <path_to_checkpoint> <encoder_npy_file> <X_test>`

For a 'new-whale', I have just found threshold which gave 28%-29%.
