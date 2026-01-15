kill_target_train_processes() {
    local TARGET_PROCESS="yibiao.zhou"
    
    # 步骤1：提取符合条件的进程PID（排除grep自身进程，避免误杀）
    # ps aux：列出所有进程
    # grep "$TARGET_PROCESS"：过滤目标进程
    # grep -v grep：排除当前grep查询进程（避免提取无效PID）
    # awk '{print $2}'：提取PID（ps aux输出第2列为PID）
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

kill_target_train_processes