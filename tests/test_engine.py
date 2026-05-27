"""
Gradient validation and computational graph testing utilities.

This script performs:
    - forward pass tracing
    - numerical gradient estimation
    - autodiff gradient computation
    - PyTorch gradient validation

NOTE: PyTorch is required to execute this script
"""

import torch
from coregrad import Scalar, draw_dot

dh = 1e-7

def h(x1, x2, x3, x4, x5, x6, x7, x8):
    """
    Construct a scalar computational graph and estimate intermediate
    gradients using finite difference approximation.
    """

    # Input Nodes
    x1 = Scalar(x1, var_name="x1")
    x2 = Scalar(x2, var_name="x2")
    x3 = Scalar(x3, var_name="x3")
    x4 = Scalar(x4, var_name="x4")
    x5 = Scalar(x5, var_name="x5")
    x6 = Scalar(x6, var_name="x6")
    x7 = Scalar(x7, var_name="x7")
    x8 = Scalar(x8, var_name="x8")

    # ----------------------------- Forward Pass -----------------------------
    z1 = x1 * x2; z1.var_name = "z1"
    z2 = z1 + x3; z2.var_name = "z2"
    z3 = z2 - x4; z3.var_name = "z3"
    z4 = z3 ** x5; z4.var_name = "z4"
    z5 = z4 / x6; z5.var_name = "z5"
    z6 = x7 / z5; z6.var_name = "z6"
    L = z6 + x8; L.var_name = "L"

    # ----------------------------- Numerical Gradients -----------------------------
    dh_local = Scalar(dh)

    # dL/dz1
    z1 = x1 * x2
    z2_h = (z1 + dh_local) + x3
    z3 = z2_h - x4
    z4 = z3 ** x5
    z5 = z4 / x6
    z6 = x7 / z5
    L_z1 = z6 + x8
    dL_dz1 = (L_z1 - L) / dh_local

    # dL/dz2
    z1 = x1 * x2
    z2 = z1 + x3
    z3_h = (z2 + dh_local) - x4
    z4 = z3_h ** x5
    z5 = z4 / x6
    z6 = x7 / z5
    L_z2 = z6 + x8
    dL_dz2 = (L_z2 - L) / dh_local

    # dL/dz3
    z1 = x1 * x2
    z2 = z1 + x3
    z3 = z2 - x4
    z4_h = (z3 + dh_local) ** x5
    z5 = z4_h / x6
    z6 = x7 / z5
    L_z3 = z6 + x8
    dL_dz3 = (L_z3 - L) / dh_local

    # dL/dz4
    z1 = x1 * x2
    z2 = z1 + x3
    z3 = z2 - x4
    z4 = z3 ** x5
    z5_h = (z4 + dh_local) / x6
    z6 = x7 / z5_h
    L_z4 = z6 + x8
    dL_dz4 = (L_z4 - L) / dh_local

    # dL/dz5
    z1 = x1 * x2
    z2 = z1 + x3
    z3 = z2 - x4
    z4 = z3 ** x5
    z5 = z4 / x6
    z6_h = x7 / (z5 + dh_local)
    L_z5 = z6_h + x8
    dL_dz5 = (L_z5 - L) / dh_local

    # dL/dz6
    z1 = x1 * x2
    z2 = z1 + x3
    z3 = z2 - x4
    z4 = z3 ** x5
    z5 = z4 / x6
    z6 = x7 / z5
    L_z6 = (z6 + dh_local) + x8
    dL_dz6 = (L_z6 - L) / dh_local

    return (
        L,
        (
            dL_dz1.data,
            dL_dz2.data,
            dL_dz3.data,
            dL_dz4.data,
            dL_dz5.data,
            dL_dz6.data,
        )
    )


if __name__ == "__main__":

    # ----------------------------- Input Values -----------------------------
    x1 = -2
    x2 = 1
    x3 = -1
    x4 = -4
    x5 = -2
    x6 = 2
    x7 = -1
    x8 = -1.2

    # ----------------------------- Forward Pass -----------------------------
    print("\n" + "=" * 100)
    print("FORWARD PASS")
    print("=" * 100)

    print("z1 = x1 * x2")
    print("z2 = z1 + x3")
    print("z3 = z2 - x4")
    print("z4 = z3 ** x5")
    print("z5 = z4 / x6")
    print("z6 = x7 / z5")
    print("L  = z6 + x8")

    L_scalar, dL_dzs = h(
        x1, x2, x3, x4,
        x5, x6, x7, x8
    )

    print("\nFinal Output:")
    print(f"L = {L_scalar.data}")

    # ----------------------------- Autograd Backward -----------------------------
    L_scalar.backward()

    print("\n" + "=" * 100)
    print("AUTOGRAD ENGINE GRADIENTS")
    print("=" * 100)

    print(f"dL/dx1 = {L_scalar._prev[0]._prev[1].grad if False else 'computed internally'}")

    inputs = [
        ("x1", x1),
        ("x2", x2),
        ("x3", x3),
        ("x4", x4),
        ("x5", x5),
        ("x6", x6),
        ("x7", x7),
        ("x8", x8),
    ]

    # ----------------------------- Numerical Gradients -----------------------------
    numerical_grads = []

    for i in range(len(inputs)):
        vals = [x1, x2, x3, x4, x5, x6, x7, x8]
        vals[i] += dh
        L_h, _ = h(*vals)
        numerical_grad = (L_h.data - L_scalar.data) / dh
        numerical_grads.append(numerical_grad)

    print("\n" + "=" * 100)
    print("NUMERICAL GRADIENTS")
    print("=" * 100)

    for (name, _), grad in zip(inputs, numerical_grads):
        print(f"dL/d{name} = {grad:.10f}")

    # ----------------------------- PyTorch Validation -----------------------------
    tx1 = torch.tensor(x1, dtype=torch.float64, requires_grad=True)
    tx2 = torch.tensor(x2, dtype=torch.float64, requires_grad=True)
    tx3 = torch.tensor(x3, dtype=torch.float64, requires_grad=True)
    tx4 = torch.tensor(x4, dtype=torch.float64, requires_grad=True)
    tx5 = torch.tensor(x5, dtype=torch.float64, requires_grad=True)
    tx6 = torch.tensor(x6, dtype=torch.float64, requires_grad=True)
    tx7 = torch.tensor(x7, dtype=torch.float64, requires_grad=True)
    tx8 = torch.tensor(x8, dtype=torch.float64, requires_grad=True)

    tz1 = tx1 * tx2
    tz2 = tz1 + tx3
    tz3 = tz2 - tx4
    tz4 = tz3 ** tx5
    tz5 = tz4 / tx6
    tz6 = tx7 / tz5
    tL = tz6 + tx8

    tL.backward()

    pytorch_grads = [
        tx1.grad.item(),
        tx2.grad.item(),
        tx3.grad.item(),
        tx4.grad.item(),
        tx5.grad.item(),
        tx6.grad.item(),
        tx7.grad.item(),
        tx8.grad.item(),
    ]

    print("\n" + "=" * 100)
    print("PYTORCH GRADIENTS")
    print("=" * 100)

    for (name, _), grad in zip(inputs, pytorch_grads):
        print(f"dL/d{name} = {grad:.10f}")

    # ----------------------------- Validation -----------------------------
    print("\n" + "=" * 100)
    print("GRADIENT VALIDATION")
    print("=" * 100)

    for (name, _), num_grad, torch_grad in zip(
        inputs,
        numerical_grads,
        pytorch_grads
    ):
        passed = abs(num_grad - torch_grad) < 1e-5
        status = "PASS" if passed else "FAIL"
        print(
            f"{name:<3} | "
            f"numerical = {num_grad:.10f} | "
            f"torch = {torch_grad:.10f} | "
            f"{status}"
        )

    # ----------------------------- Intermediate Gradients -----------------------------
    dL_dz1, dL_dz2, dL_dz3, dL_dz4, dL_dz5, dL_dz6 = dL_dzs

    print("\n" + "=" * 100)
    print("INTERMEDIATE GRADIENTS")
    print("=" * 100)

    print(f"dL/dz1 = {dL_dz1:.10f}")
    print(f"dL/dz2 = {dL_dz2:.10f}")
    print(f"dL/dz3 = {dL_dz3:.10f}")
    print(f"dL/dz4 = {dL_dz4:.10f}")
    print(f"dL/dz5 = {dL_dz5:.10f}")
    print(f"dL/dz6 = {dL_dz6:.10f}")

    # ----------------------------- Graph Visualization -----------------------------
    dot = draw_dot(L_scalar)