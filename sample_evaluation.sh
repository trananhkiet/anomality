datapath=/home/ucaokylong/anomaly_detection/patchcore-inspection/TOMO_anomaly_detect_data
#/path/to/data/from/mvtec
loadpath=/home/ucaokylong/anomaly_detection/patchcore-inspection/results/MVTecAD_Results
#/path/to/pretrained/patchcore/model

modelfolder=IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_6
#IM320_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1
# modelfolder=IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1
savefolder=evaluated_results'/'$modelfolder

datasets=( '30s' '5s' '90s' )
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python bin/load_and_evaluate_patchcore.py --gpu 3 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" \
dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath
