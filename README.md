# ONNX -> RKNN 自动转换

这个仓库提供一个 GitHub Actions 工作流，把 `best.onnx` 转成 `best.rknn`，适合 RK3568 等 Rockchip NPU 平台。

## 运行方式

### 方式一：从 Release 资产转换

1. 先在仓库的 Release 页面上传 `best.onnx`。
2. 进入 Actions，手动运行 `Convert ONNX to RKNN`。
3. 选择 `onnx_source=release`，填写 `release_tag` 和 `onnx_asset_name`。
4. 完成后，`best.rknn` 会作为 Release 资产和 Actions artifact 同时产出。

### 方式二：直接转换仓库中的 ONNX 文件

如果你把 `best.onnx` 提交到仓库根目录，也可以手动运行 workflow，选择 `onnx_source=repo`，并设置 `input_path`。

## 本地运行

```bash
python convert.py --input best.onnx --output best.rknn --target rk3568
```

如果要做量化，需要补充校准数据集：

```bash
python convert.py --input best.onnx --output best.rknn --target rk3568 --do-quantization --dataset dataset.txt
```

## 部署示例

把生成的 `best.rknn` 拷贝到开发板：

```bash
scp best.rknn pi@192.168.137.100:~/tcm_tongue/models/
```

程序可以根据 `.rknn` 文件自动启用 NPU 推理。
