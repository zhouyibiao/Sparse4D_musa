#!/bin/bash
set -euo pipefail

# ===================================== 核心配置（只需修改此处，支持平台+卡数切换） =====================================
# --- Detect hardware type ---
if command -v nvidia-smi &> /dev/null; then
    platform="nv"
elif command -v mthreads-gmi &> /dev/null; then
    platform="musa"
else
    platform="unknown"
fi
echo "[INFO] Detected hardware: $platform"

gpu_count="8"

if [ "${platform}" = "nv" ]; then
    # NVIDIA (A100) 平台基础配置
    work_space_dir="/data/yibiao.zhou/auto_driver/Sparse4D-nv"
    TRAIN_LOG_BASE_DIR="/data/yibiao.zhou/auto_driver/性能对照结果/DAF修复后性能/"

    if [ "${gpu_count}" = "1" ]; then
        export CUDA_VISIBLE_DEVICES=1
        train_tool="tools/train.py"
        log_platform_suffix="nv-单卡"
        TRAIN_LOG_DIR="${TRAIN_LOG_BASE_DIR}/NV-单卡-开箱性能_fix_overflow/"
    elif [ "${gpu_count}" = "8" ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        train_tool="tools/dist_train.sh"
        log_platform_suffix="nv-8卡"
        TRAIN_LOG_DIR="${TRAIN_LOG_BASE_DIR}/NV-8卡-开箱性能_fix_overflow/"
        dist_train_gpu_num=8  # 分布式训练卡数
    else
        echo "❌ NV平台不支持的卡数类型：${gpu_count}，仅支持 1/8"
        exit 1
    fi

elif [ "${platform}" = "musa" ]; then
    # MUSA (S5000) 平台基础配置
    work_space_dir="/data/yibiao.zhou/auto_driver/Sparse4D"
    TRAIN_LOG_BASE_DIR="/data/yibiao.zhou/auto_driver/性能优化对照结果20260109/S5000-沈伟DAF优化后性能测试结果/"
    
    if [ "${gpu_count}" = "1" ]; then
        export MUSA_VISIBLE_DEVICES=1
        train_tool="tools/train_musa.py"
        log_platform_suffix="musa-单卡"
        TRAIN_LOG_DIR="${TRAIN_LOG_BASE_DIR}"
    elif [ "${gpu_count}" = "8" ]; then
        export MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        train_tool="tools/dist_train.sh"  # 8卡分布式训练脚本
        log_platform_suffix="musa-8卡"
        TRAIN_LOG_DIR="/data/yibiao.zhou/auto_driver/性能优化对照结果20260109/S5000-沈伟DAF优化后-8卡-性能测试结果/"
        dist_train_gpu_num=8  # 分布式训练卡数
    else
        echo "❌ MUSA平台不支持的卡数类型：${gpu_count}，仅支持 1/8"
        exit 1
    fi

else
    echo "❌ 不支持的平台类型：${platform}，仅支持 musa / nv"
    exit 1
fi

CONFIG_FILE_PATH="${work_space_dir}/projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py"

CONFIG_COMBINATIONS=(
    "5 True 4"
    "5 True 8"
    "5 True 16"
    "5 True 24"
    "5 True 32"
    # "5 False 4"
    # "5 False 8"
)

# ===================================== 工具函数定义 =====================================
# 批量杀死残留训练进程
kill_target_train_processes() {
    local TARGET_PROCESS="yibiao.zhou"
    local PID_LIST=$(ps aux | grep "$TARGET_PROCESS" | grep -v grep | awk '{print $2}')
    
    if [ -n "$PID_LIST" ]; then
        echo "========================================"
        echo "找到以下目标进程PID，即将批量杀死："
        echo "$PID_LIST"
        echo "========================================"
        
        kill -9 $PID_LIST 2>/dev/null
        sleep 1
        
        echo "✅ 进程杀死命令已执行完毕"
    else
        echo "ℹ️ 未找到目标进程：$TARGET_PROCESS"
    fi
}

# ===================================== 前置检查 =====================================
# 创建日志目录
mkdir -p "${TRAIN_LOG_DIR}" || { echo "❌ 无法创建日志目录 ${TRAIN_LOG_DIR}，程序退出"; exit 1; }

# 检查工作空间目录
if [ ! -d "${work_space_dir}" ]; then
    echo "❌ 工作空间目录 ${work_space_dir} 不存在，程序退出"
    exit 1
fi

# 检查tools目录
if [ ! -d "${work_space_dir}/tools" ]; then
    echo "❌ tools目录 ${work_space_dir}/tools 不存在，程序退出"
    exit 1
fi

# 检查训练脚本是否存在（按卡数对应）
if [ ! -f "${work_space_dir}/${train_tool}" ] && [ "${train_tool##*.}" != "sh" ]; then
    echo "❌ 训练脚本 ${work_space_dir}/${train_tool} 不存在，程序退出"
    exit 1
fi
if [ "${train_tool##*.}" = "sh" ] && [ ! -f "${work_space_dir}/${train_tool}" ]; then
    echo "❌ 分布式训练脚本 ${work_space_dir}/${train_tool} 不存在，程序退出"
    exit 1
fi

# ===================================== 启动信息打印 =====================================
echo "========================================"
echo "开始批量修改配置文件并依次启动训练（抗终端断开）"
echo "当前平台：${platform}"
echo "当前卡数：${gpu_count}"
echo "工作空间目录：${work_space_dir}"
echo "配置文件路径：${CONFIG_FILE_PATH}"
echo "训练日志目录：${TRAIN_LOG_DIR}"
echo "训练脚本：${train_tool}"
echo "参数组合总数：${#CONFIG_COMBINATIONS[@]}"
echo "========================================"

# ===================================== 批量执行参数组合 =====================================
for combo in "${CONFIG_COMBINATIONS[@]}"; do
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

    # 步骤1：修改配置文件
    echo -e "\n【步骤1/2】修改配置文件..."
    python "${work_space_dir}/tools/set_bs_DAF_config.py" \
        -p "$CONFIG_FILE_PATH" \
        --config "samples_per_gpu=$bs" "num_epochs=$num_epochs" "use_deformable_func=$deformable_flag"

    # 检查配置修改是否成功，失败则跳过当前组合
    if [ $? -ne 0 ]; then
        echo "❌ 配置文件修改失败，跳过当前组合的训练流程"
        echo "--------------------------------------------------"
        continue
    fi
    echo "✅ 配置文件修改成功"

    log_filename="${log_platform_suffix}-bs-${bs}-use_deformable_func-${deformable_flag}-$(date +'%Y%m%d%H%M').log"
    full_log_path="${TRAIN_LOG_DIR}/${log_filename}"
    echo "完整日志路径：${full_log_path}"

    # 步骤3：根据「平台+卡数」启动对应的训练命令
    echo -e "\n【步骤2/2】启动训练（依次执行，完成后再进行下一个，日志：${full_log_path}）"
    echo "  训练中...（终端断开不影响，可通过 tail -f ${full_log_path} 查看实时日志）"
    
    if [ "${gpu_count}" = "1" ]; then
        nohup python3 "${work_space_dir}/${train_tool}" \
            "${CONFIG_FILE_PATH}" \
            $( [ "${platform}" = "musa" ] && echo "--enable-musa-tf32 --channel-last" ) \
            > "${full_log_path}" 2>&1
    elif [ "${gpu_count}" = "8" ]; then
        nohup bash "${work_space_dir}/${train_tool}" \
            "${CONFIG_FILE_PATH}" \
            "${dist_train_gpu_num}" \
            $( [ "${platform}" = "musa" ] && echo "--channel-last --enable-musa-tf32" ) \
            > "${full_log_path}" 2>&1
    fi

    # 检查训练执行结果
    if [ $? -eq 0 ]; then
        echo "✅ 当前组合训练完成（执行成功）"
    else
        echo "❌ 当前组合训练完成（执行失败，详见日志：${full_log_path}）"
    fi

    echo "--------------------------------------------------"
    sleep 5
done

# ===================================== 执行完毕提示 =====================================
echo -e "\n========================================"
echo "所有参数组合依次执行完毕！"
echo "当前平台：${platform}"
echo "当前卡数：${gpu_count}"
echo "所有训练日志已保存至：${TRAIN_LOG_DIR}"
echo "========================================"