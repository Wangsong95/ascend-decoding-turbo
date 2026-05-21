# 使用示例
# bash build.sh
# bash build.sh --ops=add,sub

default_ops=("add" "dynamic_quant")
# 检查是否提供了--ops参数，如果没有则使用默认值
if [[ "$@" == *"--ops"* ]]; then
    ops=$(echo "$@" | awk -F "--ops=" '{print $2}') # 使用awk提取--ops后的值，注意awk的分隔符和字段选择可能需要根据实际情况调整
else
    echo "No operations specified. Using default ops_list"
    ops="add,dynamic_quant"
fi

# 将 ops 字符串转换为数组
IFS=',' read -r -a ops_array <<< "$ops"

current_path=$(pwd)
# 遍历 ops 数组，判断每个算子是否在 default_ops 中
for op in "${ops_array[@]}"; do
    if [[ " ${default_ops[*]} " =~ " ${op} " ]]; then
        echo "✅ $op 参与此次编译"
        echo "${current_path}/ops/cce/${op}/ascend910b" >> compile.txt
    else
        echo "❌ $op 不存在于 default_ops 中，请检查"
    fi
done

mv compile.txt ./ops/cce/compile.txt
# 编译whl包
python3 -m build --wheel -n

rm -rf ./ops/cce/compile.txt