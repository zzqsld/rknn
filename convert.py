#!/usr/bin/env python3
"""ONNX -> RKNN 转换脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import numpy as np
from onnx import helper
from onnx import TensorProto


def ensure_onnx_mapping() -> None:
    if hasattr(onnx, "mapping"):
        return

    try:
        from onnx import mapping as onnx_mapping  # type: ignore

        onnx.mapping = onnx_mapping
        return
    except Exception:
        pass

    class _Mapping:
        TENSOR_TYPE_TO_NP_TYPE = {
            TensorProto.FLOAT: helper.tensor_dtype_to_np_dtype(TensorProto.FLOAT),
            TensorProto.UINT8: helper.tensor_dtype_to_np_dtype(TensorProto.UINT8),
            TensorProto.INT8: helper.tensor_dtype_to_np_dtype(TensorProto.INT8),
            TensorProto.UINT16: helper.tensor_dtype_to_np_dtype(TensorProto.UINT16),
            TensorProto.INT16: helper.tensor_dtype_to_np_dtype(TensorProto.INT16),
            TensorProto.INT32: helper.tensor_dtype_to_np_dtype(TensorProto.INT32),
            TensorProto.INT64: helper.tensor_dtype_to_np_dtype(TensorProto.INT64),
            TensorProto.BOOL: helper.tensor_dtype_to_np_dtype(TensorProto.BOOL),
            TensorProto.FLOAT16: helper.tensor_dtype_to_np_dtype(TensorProto.FLOAT16),
            TensorProto.DOUBLE: helper.tensor_dtype_to_np_dtype(TensorProto.DOUBLE),
            TensorProto.UINT32: helper.tensor_dtype_to_np_dtype(TensorProto.UINT32),
            TensorProto.UINT64: helper.tensor_dtype_to_np_dtype(TensorProto.UINT64),
        }

        NP_TYPE_TO_TENSOR_TYPE = {
            value: key for key, value in TENSOR_TYPE_TO_NP_TYPE.items()
        }

    onnx.mapping = _Mapping()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an ONNX model to RKNN")
    parser.add_argument("--input", default="best.onnx", help="Input ONNX file path")
    parser.add_argument("--output", default="best.rknn", help="Output RKNN file path")
    parser.add_argument("--target", default="rk3568", help="RKNN target platform")
    parser.add_argument(
        "--do-quantization",
        action="store_true",
        help="Enable post-training quantization",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Calibration dataset text file, required when --do-quantization is used",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_onnx_mapping()

    from rknn.api import RKNN

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    onnx_model = onnx.load(str(input_path))
    onnx.checker.check_model(onnx_model)

    print("=" * 50)
    print(f"ONNX -> RKNN conversion target: {args.target}")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print("=" * 50)

    rknn = RKNN(verbose=False)
    try:
        ret = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=args.target,
        )
        if ret != 0:
            raise RuntimeError("rknn.config failed")

        ret = rknn.load_onnx(model=str(input_path))
        if ret != 0:
            raise RuntimeError("rknn.load_onnx failed")

        build_kwargs = {"do_quantization": args.do_quantization}
        if args.do_quantization:
            if not args.dataset:
                raise ValueError("--dataset is required when --do-quantization is enabled")
            dataset_path = Path(args.dataset)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Calibration dataset not found: {dataset_path}")
            build_kwargs["dataset"] = str(dataset_path)

        ret = rknn.build(**build_kwargs)
        if ret != 0:
            raise RuntimeError("rknn.build failed")

        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError("rknn.export_rknn failed")
    finally:
        rknn.release()

    size_mb = output_path.stat().st_size / 1024 / 1024
    print()
    print("=" * 50)
    print("Conversion succeeded")
    print(f"Output size: {size_mb:.1f} MB")
    print("=" * 50)


if __name__ == "__main__":
    main()
