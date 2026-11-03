"""ResNet-18 (PyTorch) -> ONNX 변환.

설계 이유
- opset 13 고정: torch 2.4 기본값에 맡기지 않고 TRT 8.5에서 검증된 값으로 재현성 확보.
- 정적 shape (1x3x224x224): 1주차는 배치 1 고정. 동적 shape는 TRT optimization profile이
  필요해 복잡도가 올라가므로 나중에 필요할 때 추가.
- eval() 필수: BatchNorm/Dropout을 추론 모드로 고정해야 결과가 결정적.
- 변환 직후 PyTorch 출력을 저장해 두고, 나중에 TRT 출력과 비교(정확도 검증)에 사용.
"""
import argparse
import os

import numpy as np
import onnx
import torch
import torchvision


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="models/resnet18.onnx")
    p.add_argument("--opset", type=int, default=13)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    ).eval()

    torch.manual_seed(0)
    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        args.out,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["logits"],
        do_constant_folding=True,  # BN 등 상수 연산을 미리 접어 그래프 단순화
    )

    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    print(f"saved {args.out} (opset {args.opset})")
    for t in m.graph.input:
        print(" input :", t.name, [d.dim_value for d in t.type.tensor_type.shape.dim])
    for t in m.graph.output:
        print(" output:", t.name, [d.dim_value for d in t.type.tensor_type.shape.dim])

    # 정확도 비교용 기준값 저장 (TRT 결과와 나중에 비교)
    with torch.no_grad():
        ref = model(dummy).numpy()
    np.save(os.path.join(os.path.dirname(args.out), "ref_input.npy"), dummy.numpy())
    np.save(os.path.join(os.path.dirname(args.out), "ref_output.npy"), ref)
    print(" ref top-1 index:", int(ref.argmax()))


if __name__ == "__main__":
    main()
