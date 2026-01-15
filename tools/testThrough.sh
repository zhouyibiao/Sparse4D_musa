#!/bin/bash
# Usage: 
    # nv A100
    # cd /data/yibiao.zhou/auto_driver/Sparse4D-nv
    # nohup bash ./tools/testThrough.sh > /data/yibiao.zhou/auto_driver/性能对照结果/DAF修复后性能/A100_修复后性能统计.log &

    # cd /data/yibiao.zhou/auto_driver/Sparse4D
    # nohup bash ./tools/testThrough.sh > /data/yibiao.zhou/auto_driver/性能优化对照结果20260109/S5000-沈伟DAF优化后性能测试结果/S5000-沈伟DAF优化后性能测试结果.log &
    # 8卡
    # nohup bash ./tools/testThrough.sh > /data/yibiao.zhou/auto_driver/性能优化对照结果20260109/S5000-沈伟DAF优化后-8卡-性能测试结果/S5000-沈伟8卡DAF优化后性能测试结果.log &

# platform='nv'
# work_space_dir="/data/yibiao.zhou/auto_driver/Sparse4D-nv"
# CONFIG_FILE_PATH="${work_space_dir}/projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py"
# TRAIN_LOG_DIR="/data/yibiao.zhou/auto_driver/性能对照结果/DAF修复后性能/"

platform='musa'
work_space_dir="/data/yibiao.zhou/auto_driver/Sparse4D/"
CONFIG_FILE_PATH="${work_space_dir}/projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py"
TRAIN_LOG_DIR="/data/yibiao.zhou/auto_driver/性能优化对照结果20260109/S5000-沈伟DAF优化后-8卡-性能测试结果/"

# num_epoches  use_deformable_func samples_per_gpu
CONFIG_COMBINATIONS=(
    "5 True 4"
    "5 True 8"
    "5 True 16"
    "5 True 24"
    "5 True 32"
    # "5 False 4"
    # "5 False 8"
)

mkdir -p "${TRAIN_LOG_DIR}" || { echo "❌ 无法创建日志目录 ${TRAIN_LOG_DIR}，程序退出"; exit 1; }

if [ ! -d "${work_space_dir}" ]; then
    echo "❌ 工作空间目录 ${work_space_dir} 不存在，程序退出"
    exit 1
fi
if [ ! -d "${work_space_dir}/tools" ]; then
    echo "❌ tools目录 ${work_space_dir}/tools 不存在，程序退出"
    exit 1
fi

echo "========================================"
echo "开始批量修改配置文件并依次启动训练（抗终端断开）"
echo "工作空间目录：${work_space_dir}"
echo "配置文件路径：$CONFIG_FILE_PATH"
echo "训练日志目录：$TRAIN_LOG_DIR"
echo "参数组合总数：${#CONFIG_COMBINATIONS[@]}"
echo "提示：脚本已启用nohup保护，终端/远程断开后训练仍会继续"
echo "========================================"

kill_target_train_processes() {
    local TARGET_PROCESS="yibiao.zhou"
    
    local PID_LIST=$(ps aux | grep "$TARGET_PROCESS" | grep -v grep | awk '{print $2}')
    
    if [ -n "$PID_LIST" ]; then
        echo "========================================"
        echo "找到以下目标进程PID，即将批量杀死："
        echo "$PID_LIST"
        echo "========================================"
        
        kill -9 $PID_LIST
        sleep 1
        
        echo "✅ 进程杀死命令已执行完毕"
    else
        echo "ℹ️ 未找到目标进程：$TARGET_PROCESS"
    fi
}

for combo in "${CONFIG_COMBINATIONS[@]}"
do
    echo "开始清理残留的训练进程..."
    kill_target_train_processes
    kill_target_train_processes
    echo "残留进程清理完成，继续后续流程..."
    read -r num_epochs deformable_flag bs <<< "$combo"

    echo -e "\n=================================================="
    echo "当前执行参数组合："
    echo "  num_epochs=$num_epochs"
    echo "  use_deformable_func=$deformable_flag"
    echo "  samples_per_gpu=$bs"
    echo "=================================================="

    # 第一步：修改配置文件（单独检查执行状态，使用work_space_dir下的tools绝对路径）
    echo -e "\n【步骤1/2】修改配置文件..."
    python "${work_space_dir}/tools/set_bs_DAF_config.py" \
        -p "$CONFIG_FILE_PATH" \
        --config "samples_per_gpu=$bs" "num_epochs=$num_epochs" "use_deformable_func=$deformable_flag"

    # 检查配置修改是否成功，失败则跳过当前组合的训练
    if [ $? -ne 0 ]; then
        echo "❌ 配置文件修改失败，跳过当前组合的训练流程"
        echo "--------------------------------------------------"
        continue
    fi
    echo "✅ 配置文件修改成功"

    # 第二步：构造日志文件名（使用正确变量，避免空值）
    # log_filename="nv-DAF_bs-${bs}-use_deformable_func-${deformable_flag}-num_workers-8-$(date +'%Y%m%d%H%M').log"
    # log_filename="musa-单卡-DAF_bs-${bs}-use_deformable_func-${deformable_flag}-num_workers-8-$(date +'%Y%m%d%H%M').log"
    log_filename="musa-8卡-DAF_bs-${bs}-use_deformable_func-${deformable_flag}-num_workers-8-$(date +'%Y%m%d%H%M').log"
    full_log_path="${TRAIN_LOG_DIR}/${log_filename}"
    echo "full_log_path: ${full_log_path}"

    echo -e "\n【步骤2/2】启动训练（依次执行，完成后再进行下一个，日志：${full_log_path}）"
    echo "  训练中...（终端断开不影响，可通过 tail -f ${full_log_path} 查看实时日志）"
    
    # nohup 包裹训练命令（抗SIGHUP信号，终端断开后训练继续），前台执行（无&）实现依次执行
    # CUDA_VISIBLE_DEVICES=1 nohup python3 "${work_space_dir}/tools/train.py" \
    #     "${CONFIG_FILE_PATH}" \
    #     > "${full_log_path}" 2>&1
    
    # musa 单卡
    # MUSA_VISIBLE_DEVICES=1 nohup python3 "${work_space_dir}/tools/train_musa.py" \
    #     "${CONFIG_FILE_PATH}" \
    #     --enable-musa-tf32 \
    #     --channel-last \
    #     > "${full_log_path}" 2>&1

    # musa 8卡
    MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash "${work_space_dir}/tools/dist_train.sh" \
        /data/yibiao.zhou/auto_driver/Sparse4D/projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
        8 \
        --channel-last  --enable-musa-tf32 \
        > "${full_log_path}" 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ 当前组合训练完成（执行成功）"
    else
        echo "❌ 当前组合训练完成（执行失败，详见日志：${full_log_path}）"
    fi

    echo "--------------------------------------------------"
    sleep 5
done

# 所有组合执行完毕提示
echo -e "\n========================================"
echo "所有参数组合依次执行完毕！"
echo "所有训练日志已保存至：${TRAIN_LOG_DIR}"
echo "========================================"