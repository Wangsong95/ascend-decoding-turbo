current_path=$(pwd)

# 清理编译缓存内容
clean(){
    rm -rf ${current_path}/adt_triton_ops.egg-info
    rm -rf ${current_path}/build
    rm -rf ${current_path}/dist
    rm -rf ${current_path}/build_whl.log
    rm -rf ${current_path}/adt_triton_ops
}
# 创建编译目录
mkdir ${current_path}/adt_triton_ops
cp *.py ${current_path}/adt_triton_ops
rm -rf ${current_path}/adt_triton_ops/setup.py
# 编译出adt_triton_ops算子包
python setup.py bdist_wheel > build_whl.log
# 安装adt_triton_ops算子包
pip install --force-reinstall --no-deps ${current_path}/dist/adt_triton_ops-0.1.0-py3-none-any.whl
clean