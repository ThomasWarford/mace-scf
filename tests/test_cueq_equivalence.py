"""cuequivariance parity: a cueq-enabled model must match its e3nn twin.

Adding a model type: append a ``ModelCase`` to ``CASES``. A model whose build path has
no cuequivariance support raises ``NotImplementedError`` in ``build_model`` and is
skipped here, so its cases start running by themselves once support lands.

The parity checks need CUDA (cuequivariance's symmetric contraction and layout
transpose have no CPU kernels); the configuration checks run anywhere. Forward parity
covers forces and stress; the gradient test backpropagates through both, so the
double-backward paths are exercised in float64.
"""

import importlib.util
import os
from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np
import pytest
import torch

import mace.cli.convert_e3nn_cueq as e3nn_cueq
import mace.tools
import mace_scf.utils
from mace_scf.utils.check_args import check_config_conflicts
from mace_scf.utils.model_training_wrappers import make_model_wrapper
from mace_scf.utils.run_train_utils import (
    build_model,
    get_atom_density_scaling,
    get_fermi_level_offset,
    get_field_feature_norms,
    get_formal_charges,
)
from tests.utils import dataset_from_atoms, seed_torch, water_configs

CUET_AVAILABLE = importlib.util.find_spec("cuequivariance_torch") is not None
TORCHOPT_AVAILABLE = importlib.util.find_spec("torchopt") is not None
Z_TABLE = mace.tools.get_atomic_number_table_from_zs([1, 8])  # water_configs()
ATOL = float(os.environ.get("CUEQ_ATOL", "1e-9"))
RTOL = float(os.environ.get("CUEQ_RTOL", "1e-7"))

HEADS = (
    '{"default": {"info_keys": {"energy": "REF_energy", "total_charge": "total_charge"},'
    ' "arrays_keys": {"forces": "REF_forces"}}}'
)
SCHEDULE = (
    '{0: {"name": "stage1", "start": 0, "end": 1,'
    ' "loss": {"energy_per_atom": 1.0, "forces": 10.0}, "lr": 0.01}}'
)

requires_cuet = pytest.mark.skipif(
    not CUET_AVAILABLE, reason="cuequivariance_torch is not installed"
)
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="cuequivariance symmetric contraction / layout transpose are CUDA-only",
)


@pytest.fixture(autouse=True)
def float64_default_dtype():
    """Every model in this file is built in float64, the configuration tests' included.

    Building a cueq model under pytest's float32 default first leaves state behind that the
    later float64 pairs pick up: e3nn and cueq then disagree by ~1e-9 (rel ~1e-6) instead
    of ~1e-14, which reads as tolerance noise but is not. The dtype is restored after.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@dataclass(frozen=True)
class ModelCase:
    """One --model value plus the flags it needs and the outputs worth comparing."""

    model: str
    extra_argv: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ("energy", "forces")
    name: str = ""  # test id; defaults to the model and fixed-point mode
    # FixedPoint only: the training mode its stage, and so its FixedPointWrapper, runs in
    fixed_point_mode: str = ""

    @property
    def id(self) -> str:
        return self.name or "-".join(filter(None, (self.model, self.fixed_point_mode)))


ELECTROSTATIC_ARGV = (
    "--electrostatic_pbc_method", "pbc",
    "--atomic_multipoles_max_l", "1",
    "--atomic_multipoles_smearing_width", "1.5",
    "--kspace_cutoff_factor", "0.75",
)
# water is H/O, so every charged model needs formal charges for both
FORMAL_CHARGES_ARGV = ("--atomic_formal_charges", "{1: 1.0, 8: -2.0}")
NONLINEAR_ZBL_AGNESI_ARGV = (
    "--interaction_first", "RealAgnosticResidualNonLinearInteractionBlock",
    "--interaction", "RealAgnosticResidualNonLinearInteractionBlock",
    "--pair_repulsion",
    "--distance_transform", "Agnesi",
)
POLAR_BACKBONE_ARGV = NONLINEAR_ZBL_AGNESI_ARGV + (
    "--hidden_irreps", "8x0e + 8x1o",
    "--edge_irreps", "4x0e + 4x1o",
    "--use_agnostic_product", "True",
)

# FixedPoint has no forward of its own; the parity tests drive it through FixedPointWrapper
# in each supported mode. linearize_solve is refused under --enable_cueq (check_args).
FIXED_POINT_OUTPUTS = ("energy", "forces", "density_coefficients", "dipole", "fermi_level")
# run_train scans the data for these when unset; pin them (non-trivially) instead. Two
# field widths and field_feature_max_l = atomic_multipoles_max_l = 1 make four norms.
FIXED_POINT_ARGV = ELECTROSTATIC_ARGV + (
    "--fermi_level_offset", "-0.5",
    "--field_feature_norms", "[0.8, 1.2, 0.9, 1.1]",
)

CASES = [
    ModelCase(
        "LocalSplitCharges",
        FORMAL_CHARGES_ARGV + ELECTROSTATIC_ARGV,
        ("energy", "forces", "stress", "density_coefficients", "dipole"),
    ),
    ModelCase("LocalCharges", ELECTROSTATIC_ARGV,
              ("energy", "forces", "stress", "density_coefficients")),
    ModelCase(
        "FixedChargeBaselinedMACE",
        FORMAL_CHARGES_ARGV + ELECTROSTATIC_ARGV,
        ("energy", "forces", "stress"),
    ),
    ModelCase("MACE", (), ("energy", "forces", "stress")),
    # the MACE-Polar / OMol backbone: NonLinear blocks, node width decoupled from the
    # edge width, element-agnostic product, plus 0b3's ZBL + Agnesi. Hidden and edge
    # widths differ so a dropped --edge_irreps cannot pass by coincidence.
    ModelCase("MACE", POLAR_BACKBONE_ARGV, ("energy", "forces", "stress"),
              name="MACE-polar-backbone"),
    # the same backbone under LocalSplitCharges, with ScaleShiftMACE's scale/shift, as the
    # R2-backbone LSC fits run it
    ModelCase(
        "LocalSplitCharges",
        FORMAL_CHARGES_ARGV + ELECTROSTATIC_ARGV + POLAR_BACKBONE_ARGV
        + ("--local_scale_shift", "True", "--mean", "-1.5", "--std", "0.7"),
        ("energy", "forces", "stress", "density_coefficients", "dipole"),
        name="LocalSplitCharges-polar-backbone",
    ),
    *[
        ModelCase("FixedPoint", FIXED_POINT_ARGV, FIXED_POINT_OUTPUTS, fixed_point_mode=mode)
        for mode in ("direct", "unroll_scf", "implicit")
    ],
    # NonLinear blocks, so the mul_ir Gate-transpose fix is exercised (FixedPoint has no
    # --edge_irreps / --use_agnostic_product). The backbone runs once, before the SCF, so
    # "direct" covers it; the SCF modes are covered above.
    ModelCase(
        "FixedPoint",
        FIXED_POINT_ARGV + NONLINEAR_ZBL_AGNESI_ARGV,
        FIXED_POINT_OUTPUTS,
        name="FixedPoint-polar-backbone",
        fixed_point_mode="direct",
    ),
]
CASE_IDS = [case.id for case in CASES]
# the configuration tests only need each distinct model once; FixedPoint's mode is a
# training option and does not change what build_model returns
BUILD_CASES = [c for c in CASES if c.fixed_point_mode in ("", "direct")]
BUILD_CASE_IDS = [case.id for case in BUILD_CASES]


def _schedule(fixed_point_mode: str) -> str:
    if not fixed_point_mode:
        return SCHEDULE
    options = {"mode": fixed_point_mode}
    if fixed_point_mode != "direct":
        options["scf"] = {
            "num_scf_steps": 40,
            "scf_tolerance": 1e-8,
            "mixing_parameter": 0.5,
            "constant_charge": True,
            "initial_density": "from_data",
            "initial_fermi_level": "from_data",
        }
    return (
        '{0: {"name": "stage1", "start": 0, "end": 1,'
        ' "loss": {"energy_per_atom": 1.0, "forces": 10.0}, "lr": 0.01,'
        f' "fixed_point_training_options": {options!r}}}}}'
    )


def _args(case: ModelCase, enable_cueq: bool, device: str):
    """Parse a minimal but real argv, as run_train would see it."""
    if case.fixed_point_mode == "implicit" and not TORCHOPT_AVAILABLE:
        pytest.skip("implicit mode needs torchopt")  # even to parse the schedule
    argv = [
        "--name", "cueq_test",
        "--train_file", "unused.xyz",  # only the suffix is checked; never read here
        "--heads", HEADS,
        "--train_schedule", _schedule(case.fixed_point_mode),
        "--error_table", "PerAtomRMSE",
        "--model", case.model,
        "--hidden_irreps", "4x0e + 4x1o",
        "--MLP_irreps", "4x0e",
        "--r_max", "3.0",
        "--max_ell", "3",
        "--correlation", "3",
        "--num_interactions", "2",
        "--compute_avg_num_neighbors", "False",
        "--avg_num_neighbors", "10.0",
        "--compute_polarizability", "False",
        "--default_dtype", "float64",
        "--device", device,
        "--enable_cueq", str(enable_cueq),
        *case.extra_argv,
    ]
    args = mace_scf.utils.extended_arg_parser().parse_args(argv)
    check_config_conflicts(args)
    # what run_train resolves next from the data; every value here is explicit, so the
    # (absent) loader is never read
    args.fermi_level_offset = get_fermi_level_offset(None, args, device)
    args.field_feature_norms = get_field_feature_norms(
        None, args, device, fermi_level_offset=args.fermi_level_offset
    )
    args.atom_density_scaling = get_atom_density_scaling(None, args, device, Z_TABLE)
    return args


def _build(args):
    return build_model(
        args,
        Z_TABLE,
        np.zeros(len(Z_TABLE)),
        get_formal_charges(
            args.model, args.formal_charges_from_data, args.atomic_formal_charges, Z_TABLE
        ),
        train_loader=None,
    )


def _run(model, case: ModelCase, args, batch, training: bool):
    """Forward through the wrapper run_train puts around the model for this stage."""
    wrapper = make_model_wrapper(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        output_args={"forces": True, "virials": False, "stress": "stress" in case.outputs},
        fixed_point_training_options=args.train_schedule[0].get("fixed_point_training_options"),
    )
    return wrapper(batch.to_dict(), training=training)


def _cueq_config_of(model):
    """The CuEquivarianceConfig the MACE blocks were built with, if any."""
    for module in model.modules():
        cfg = getattr(module, "cueq_config", None)
        if cfg is not None:
            return cfg
    return None


def transfer_e3nn_to_cueq(source, target, correlation: int) -> None:
    """Copy weights from an e3nn model into its cuequivariance twin.

    Two shapes of the problem, both handled:

    * the symmetric contraction is a cueq kernel -- it stores one weight per product
      block where e3nn stores several, so upstream's transfer does the concatenation.
      Its ``get_kmax_pairs`` assumes MACE's scalar-only last layer, which is wrong for
      models that keep full hidden irreps there (LocalSplitCharges), so the number of
      contractions is read off the source model instead and works for both;
    * the symmetric contraction stayed e3nn (``optimize_symmetric`` off) -- the tensors
      line up by name, but upstream would still write cueq-format keys, so copy directly.

    ``use_reduced_cg`` is False because every model here builds the original MACE basis:
    mace_scf's own models leave it unset, so the product block forwards ``None``, and the
    MACE branch passes ``args.use_reduced_cg``, whose command-line default is False.
    e3nn then builds the original basis and cueq gets ``original_mace=True``, so no basis
    projection is needed. ``_build_pair`` asserts that this still holds.
    """
    target_sd = target.state_dict()
    source_sd = source.state_dict()

    if not any(k.endswith("symmetric_contractions.weight") for k in target_sd):
        merged = dict(target_sd)
        for key, value in source_sd.items():
            if key not in merged:
                continue
            if value.shape == merged[key].shape:
                merged[key] = value
            elif e3nn_cueq.shapes_match_up_to_unsqueeze(value.shape, merged[key].shape):
                merged[key] = e3nn_cueq.reshape_like(value, merged[key].shape)
            else:
                raise ValueError(
                    f"cannot map {key}: e3nn {tuple(value.shape)} vs cueq {tuple(merged[key].shape)}"
                )
        unmatched = sorted(
            (set(target_sd) - set(source_sd)) & {n for n, _ in target.named_parameters()}
        )
        if unmatched:
            raise ValueError(f"cueq parameters with no e3nn counterpart: {unmatched[:10]}")
        target.load_state_dict(merged, strict=True)
        return

    kmax_pairs = [
        [i, len(product.symmetric_contractions.contractions) - 1]
        for i, product in enumerate(source.products)
    ]
    original = e3nn_cueq.get_kmax_pairs
    e3nn_cueq.get_kmax_pairs = lambda *_args, **_kwargs: kmax_pairs
    try:
        e3nn_cueq.transfer_weights(
            source,
            target,
            num_product_irreps=max(k for _, k in kmax_pairs),
            correlation=correlation,
            num_layers=len(source.interactions),
            use_reduced_cg=False,
        )
    finally:
        e3nn_cueq.get_kmax_pairs = original


def contraction_grad(grads, layer: int, kmax: int) -> torch.Tensor:
    """e3nn contraction gradients, concatenated the way the weights are."""
    return torch.cat(
        [
            grads[f"products.{layer}.symmetric_contractions.contractions.{k}.weights{suffix}"]
            for k in range(kmax + 1)
            for suffix in ("_max", ".0", ".1")
        ],
        dim=1,
    )


def _build_pair(case: ModelCase, device: str):
    """An e3nn model and its cueq twin holding identical weights."""
    torch.set_default_dtype(torch.float64)
    atoms = water_configs()
    args_e3nn = _args(case, False, device)

    seed_torch(0)
    e3nn_model = _build(args_e3nn)
    # the weight transfer and contraction_grad both assume the original MACE CG basis
    assert not getattr(e3nn_model, "use_reduced_cg", False), (
        f"{case.model} built the reduced CG basis; the parity transfer would need "
        "symmetric_contraction_proj on both the weights and the gradients"
    )
    try:
        cueq_model = _build(_args(case, True, device))
    except NotImplementedError as exc:
        pytest.skip(f"{case.model} has no cuequivariance support yet: {exc}")

    e3nn_model, cueq_model = e3nn_model.to(device), cueq_model.to(device)
    transfer_e3nn_to_cueq(e3nn_model, cueq_model, correlation=args_e3nn.correlation)
    return e3nn_model, cueq_model, args_e3nn, atoms


def _with_scf_inputs(atoms):
    """Copies carrying the reference density, Fermi level and field FixedPoint reads.

    Left unset these all default to zero, which would make "direct" compare the field
    update at a trivial input.
    """
    rng = np.random.default_rng(0)
    out = []
    for at in atoms:
        at = at.copy()
        at.arrays["REF_multipoles"] = 0.1 * rng.standard_normal((len(at), 4))
        at.info["REF_fermi_level"] = -0.7
        at.info["REF_external_field"] = np.array([0.02, -0.01, 0.03])
        out.append(at)
    return out


def _batch(atoms, device: str, n: int = 2, scf_inputs: bool = False):
    atoms = atoms[:n]
    kwargs = {}
    if scf_inputs:
        atoms = _with_scf_inputs(atoms)
        kwargs = dict(
            atomic_multipoles_key="REF_multipoles",
            fermi_level_key="REF_fermi_level",
            external_field_key="REF_external_field",
        )
    dataset = dataset_from_atoms(atoms, cutoff=3.0, atomic_multipoles_max_l=1, **kwargs)
    loader = mace.tools.torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=n, shuffle=False
    )
    return next(iter(loader)).to(device)


@pytest.mark.cueq
@requires_cuet
@pytest.mark.parametrize("case", BUILD_CASES, ids=BUILD_CASE_IDS)
def test_enable_cueq_is_never_silently_ignored(case):
    """--enable_cueq must either build cueq modules or refuse; never a plain e3nn model."""
    try:
        model = _build(_args(case, True, "cpu"))
    except NotImplementedError:
        pytest.skip(f"{case.model} has no cuequivariance support yet")
    assert any(
        type(m).__module__.startswith("cuequivariance_torch") for m in model.modules()
    ), f"{case.model}: --enable_cueq True produced no cuequivariance modules"


@pytest.mark.cueq
@requires_cuet
@pytest.mark.parametrize("case", BUILD_CASES, ids=BUILD_CASE_IDS)
def test_cueq_config_is_safe(case):
    """Guard the two settings that silently corrupt this model family.

    conv_fusion: MACE builds the fused kernel without passing `layout`, so it works in
    ir_mul while the surrounding linears use the configured layout -- the interaction
    messages come out wrong (~2 absolute) while every other block matches to 1e-15.
    layout: node features leave the MACE blocks and feed mace_scf's own e3nn blocks,
    which assume e3nn's mul_ir ordering.
    """
    try:
        model = _build(_args(case, True, "cpu"))
    except NotImplementedError:
        pytest.skip(f"{case.model} has no cuequivariance support yet")
    cfg = _cueq_config_of(model)
    assert cfg is not None and cfg.enabled, f"{case.model}: no CuEquivarianceConfig reached the blocks"
    assert cfg.conv_fusion is False, "conv_fusion corrupts interaction messages for these models"
    assert cfg.layout_str == "mul_ir", "mace_scf's own e3nn blocks require the mul_ir layout"


@pytest.mark.cueq
@requires_cuet
@requires_gpu
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_forward_matches_e3nn(case):
    e3nn_model, cueq_model, args, atoms = _build_pair(case, "cuda")
    batch = _batch(atoms, "cuda", scf_inputs=bool(case.fixed_point_mode))
    e3nn_model.eval()
    cueq_model.eval()
    reference = _run(e3nn_model, case, args, batch, training=False)
    actual = _run(cueq_model, case, args, batch, training=False)
    if case.fixed_point_mode not in ("", "direct"):
        # a diverging SCF matches only in noise; the fixture must converge to mean anything
        history = reference["charges_history"]
        assert (history[..., -1] - history[..., -2]).abs().max() < 1e-6, "e3nn SCF did not converge"
    for key in case.outputs:
        assert reference.get(key) is not None, f"{case.model}: e3nn produced no {key}"
        torch.testing.assert_close(
            actual[key], reference[key], atol=ATOL, rtol=RTOL, msg=lambda m, k=key: f"{case.model} {k}: {m}"
        )


@pytest.mark.cueq
@requires_cuet
@requires_gpu
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_gradients_match_e3nn(case):
    e3nn_model, cueq_model, args, atoms = _build_pair(case, "cuda")
    batch = _batch(atoms, "cuda", scf_inputs=bool(case.fixed_point_mode))

    def grads(model):
        for param in model.parameters():
            param.grad = None
        model.train()
        out = _run(model, case, args, batch, training=True)
        loss = out["energy"].sum() + out["forces"].pow(2).sum()
        if out.get("stress") is not None:
            loss = loss + out["stress"].pow(2).sum()
        loss.backward()
        return {n: p.grad.detach() for n, p in model.named_parameters() if p.grad is not None}

    reference, actual = grads(e3nn_model), grads(cueq_model)

    compared = 0
    for name, ref in reference.items():
        if name not in actual:
            continue  # reparameterised tensors are handled below
        got = actual[name]
        if ref.shape != got.shape:
            # cueq linears keep a leading singleton dimension
            assert ref.numel() == got.numel(), f"{case.model} {name}: {ref.shape} vs {got.shape}"
            ref, got = ref.flatten(), got.flatten()
        torch.testing.assert_close(got, ref, atol=ATOL, rtol=RTOL,
                                   msg=lambda m, n=name: f"{case.model} {n}: {m}")
        compared += 1

    # symmetric contractions: same gradient, different parameterisation
    for layer, product in enumerate(getattr(e3nn_model, "products", [])):
        key = f"products.{layer}.symmetric_contractions.weight"
        if key not in actual:
            continue
        expected = contraction_grad(reference, layer, len(product.symmetric_contractions.contractions) - 1)
        torch.testing.assert_close(actual[key], expected, atol=ATOL, rtol=RTOL,
                                   msg=lambda m, k=key: f"{case.model} {k}: {m}")
        compared += 1

    assert compared > 0, f"{case.model}: no gradients were compared"


@pytest.mark.cueq
@requires_cuet
@requires_gpu
def test_nonlinear_gate_transposes_break_parity_under_mul_ir():
    """Pins why build_model drops the NonLinear block's Gate transposes: put them back
    and the cueq model stops matching e3nn. If upstream ever makes them layout-aware
    this test fails, and _disable_mul_ir_nonlinear_transposes can go."""
    from mace.modules.blocks import RealAgnosticResidualNonLinearInteractionBlock
    from mace.modules.wrapper_ops import TransposeIrrepsLayoutWrapper

    case = next(c for c in CASES if c.id == "MACE-polar-backbone")
    e3nn_model, cueq_model, _, atoms = _build_pair(case, "cuda")
    restored = 0
    for block in cueq_model.modules():
        if isinstance(block, RealAgnosticResidualNonLinearInteractionBlock):
            assert block.transpose_mul_ir is None and block.transpose_ir_mul is None
            block.transpose_mul_ir = TransposeIrrepsLayoutWrapper(
                irreps=block.irreps_nonlin, source="ir_mul", target="mul_ir",
                cueq_config=block.cueq_config,
            ).to("cuda")
            block.transpose_ir_mul = TransposeIrrepsLayoutWrapper(
                irreps=block.irreps_out, source="mul_ir", target="ir_mul",
                cueq_config=block.cueq_config,
            ).to("cuda")
            restored += 1
    assert restored == 2

    batch = _batch(atoms, "cuda")
    e3nn_model.eval()
    cueq_model.eval()
    reference = e3nn_model(batch.to_dict(), training=False, compute_force=True)
    actual = cueq_model(batch.to_dict(), training=False, compute_force=True)
    assert not torch.allclose(actual["forces"], reference["forces"], atol=1e-6, rtol=0)


@pytest.mark.cueq
def test_fixed_point_linearize_solve_is_refused_under_cueq():
    case = replace(next(c for c in CASES if c.model == "FixedPoint"), fixed_point_mode="linearize_solve")
    _args(case, False, "cpu")  # fine without cueq
    with pytest.raises(NotImplementedError, match="linearize_solve"):
        _args(case, True, "cpu")
