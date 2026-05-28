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
            TensorProto.FLOAT: np.float32,
            TensorProto.UINT8: np.uint8,
            TensorProto.INT8: np.int8,
            TensorProto.UINT16: np.uint16,
            TensorProto.INT16: np.int16,
            TensorProto.INT32: np.int32,
            TensorProto.INT64: np.int64,
            TensorProto.BOOL: "bool",
            TensorProto.FLOAT16: np.float16,
            TensorProto.DOUBLE: np.float64,
            TensorProto.UINT32: np.uint32,
            TensorProto.UINT64: np.uint64,
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


def rewrite_input_as_uint8(model_path: Path) -> Path:
    model = onnx.load(str(model_path))
    graph = model.graph

    if not graph.input:
        raise ValueError("ONNX model has no inputs")

    input_name = graph.input[0].name
    graph.input[0].type.tensor_type.elem_type = TensorProto.UINT8

    cast_node = helper.make_node(
        "Cast",
        inputs=[input_name],
        outputs=["cast_out"],
        to=TensorProto.FLOAT,
    )
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [1], [255.0])
    div_node = helper.make_node(
        "Div",
        inputs=["cast_out", "scale"],
        outputs=["preprocessed"],
    )

    for node in graph.node:
        for index, input_tensor in enumerate(node.input):
            if input_tensor == input_name:
                node.input[index] = "preprocessed"

    graph.node.insert(0, div_node)
    graph.node.insert(0, cast_node)
    graph.initializer.append(scale)

    fixed_path = model_path.with_name(f"{model_path.stem}_fixed.onnx")
    onnx.save(model, str(fixed_path))
    onnx.checker.check_model(onnx.load(str(fixed_path)))
    return fixed_path


def main() -> None:
    args = parse_args()

    ensure_onnx_mapping()

    from rknn.api import RKNN

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    fixed_input_path = rewrite_input_as_uint8(input_path)

    onnx_model = onnx.load(str(fixed_input_path))
    onnx.checker.check_model(onnx_model)

    print("=" * 50)
    print(f"ONNX -> RKNN conversion target: {args.target}")
    print(f"Input : {fixed_input_path}")
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

        ret = rknn.load_onnx(model=str(fixed_input_path))
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
