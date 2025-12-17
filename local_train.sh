if command -v nvidia-smi &> /dev/null; then
    HW="CUDA"
elif command -v mthreads-gmi &> /dev/null; then
    HW="MUSA"
else
    echo "No supported GPU hardware found. Exiting."
    exit 1
fi
echo "Detected hardware: ${HW}"

case $HW in
    "CUDA")
        export CUDA_VISIBLE_DEVICES=0
        ;;
    "MUSA")
        export MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        ;;
    *)
        echo "Unsupported hardware: ${HW}. Exiting."
        exit 1
        ;;
esac

export PYTHONPATH=$PYTHONPATH:./
VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$MUSA_VISIBLE_DEVICES}
gpus=(${VISIBLE_DEVICES//,/ })
gpu_num=${#gpus[@]}
echo "number of gpus: "${gpu_num}

config=projects/configs/$1.py

if [ ${gpu_num} -gt 1 ]
then
    bash ./tools/dist_train.sh \
        ${config} \
        ${gpu_num} \
        --work-dir=work_dirs/$1 \
        --channel-last  --enable-musa-tf32
else
    if [ "$HW" == "MUSA" ]; then
        python ./tools/train_musa.py \
            ${config} \
            --channel-last  --enable-musa-tf32

        exit 0
    elif [ "$HW" == "CUDA" ]; then
        python ./tools/train.py \
        ${config}
    else
        echo "Unsupported hardware: ${HW}. Exiting."
        exit 1
    fi
    
fi
