"""Dependency-free checks, runnable with `python -m unittest hybrid.tests.test_config`."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from dataclasses import asdict, replace

from hybrid.config import (PRELUDE_MODES, Config, default_run_name, execution_plan,
                           lr_multiplier, normalize_zo_settings, parse_args, recipe, transition_fraction)


class RecipeTests(unittest.TestCase):
    def test_agreed_configuration(self):
        plan = recipe(parse_args([]), 8)
        self.assertEqual(plan["total_parameters"], 218786816)
        self.assertEqual(plan["effective_depth"], 20)
        self.assertEqual(plan["num_iterations"], 19523)
        self.assertEqual(plan["actual_tokens"], 5117837312)
        self.assertEqual(plan["grad_accum_steps"], 8)
        self.assertAlmostEqual(plan["initial_lrs"]["matrix"], 0.007074509700636991)

    def test_three_modes_share_recipe(self):
        plans = [recipe(Config(mode=mode), 8) for mode in ("full_fo", "truncated", "hybrid")]
        self.assertEqual(plans[0], plans[1])
        self.assertEqual(plans[1], plans[2])

    def test_prelude_variants_share_recipe_and_have_distinct_outputs(self):
        configs = [Config(mode="hybrid", prelude_mode=mode) for mode in PRELUDE_MODES]
        for cfg in configs:
            self.assertEqual(recipe(cfg, 8), recipe(configs[0], 8))
        self.assertEqual(len({default_run_name(cfg) for cfg in configs}), 4)
        self.assertEqual(default_run_name(configs[0]), "loop2_s16_hybrid_seed42")

    def test_prelude_execution_plans(self):
        for mode in PRELUDE_MODES:
            cfg = Config(mode="hybrid", prelude_mode=mode, zo_directions=2)
            plan = execution_plan(cfg)
            self.assertEqual(plan["parameter_or_joint_probe_pairs"], 2)
            self.assertEqual(plan["input_only_probe_pairs"], 2 if mode == "separate" else 0)
            self.assertEqual(plan["extra_prelude_forwards"], int(mode == "joint_recompute"))
            self.assertEqual(plan["prelude_backwards"], 1)
            cfg.zo_lambda = 0
            plan = execution_plan(cfg)
            self.assertEqual(plan["parameter_or_joint_probe_pairs"], 0)
            self.assertEqual(plan["input_only_probe_pairs"], 0)
            self.assertEqual(plan["extra_prelude_forwards"], 0)

    def test_invalid_prelude_options(self):
        for cfg in (Config(prelude_mode="joint"), Config(mode="hybrid", prelude_mode="typo"),
                    Config(mode="hybrid", zo_rho=0), Config(mode="hybrid", zo_rho=float("nan")),
                    Config(mode="hybrid", zo_rho=1e308, zo_epsilon=10)):
            with self.assertRaises(ValueError):
                cfg.validate()

    def test_resume_preserves_prelude_settings_and_accepts_old_defaults(self):
        from hybrid.train import validate_resume
        for mode in PRELUDE_MODES:
            cfg = Config(mode="hybrid", prelude_mode=mode)
            meta = {"world_size": 8, "user_config": asdict(cfg)}
            validate_resume(replace(cfg, resume_step=100, stop_after=200), meta, 8)
            for changed in (replace(cfg, zo_rho=2),
                            replace(cfg, prelude_mode="joint" if mode != "joint" else "none")):
                with self.assertRaisesRegex(ValueError, "Resume changes experimental setting"):
                    validate_resume(changed, meta, 8)
        cfg = Config(mode="hybrid")
        old = asdict(cfg)
        del old["prelude_mode"], old["zo_rho"]
        validate_resume(cfg, {"world_size": 8, "user_config": old}, 8)
        old["zo_input_epsilon"] = 2e-3
        validate_resume(replace(cfg, zo_rho=2), {"world_size": 8, "user_config": old}, 8)

    def test_pilot_does_not_change_lr_or_horizon(self):
        self.assertEqual(recipe(Config(stop_after=100), 8), recipe(Config(), 8))

    def test_norm_matching_configuration_and_resume(self):
        from hybrid.train import validate_resume
        cfg = parse_args(["--mode", "hybrid", "--zo-norm-ratio", "0.3"])
        self.assertEqual(cfg.zo_norm_ratio, 0.3)
        self.assertEqual(recipe(cfg, 8), recipe(Config(), 8))
        self.assertIn("_norm0.3_", default_run_name(cfg))
        meta = {"world_size": 8, "user_config": asdict(cfg)}
        validate_resume(replace(cfg, resume_step=2000), meta, 8)
        with self.assertRaisesRegex(ValueError, "zo_norm_ratio"):
            validate_resume(replace(cfg, zo_norm_ratio=1), meta, 8)
        old = asdict(Config())
        del old["zo_norm_ratio"]
        validate_resume(Config(), {"world_size": 8, "user_config": old}, 8)
        for invalid in (replace(cfg, zo_norm_ratio=-1), replace(cfg, zo_norm_ratio=float("nan")),
                        replace(cfg, mode="full_fo"), replace(cfg, prelude_mode="joint"),
                        replace(cfg, zo_lambda=0.1)):
            with self.assertRaises(ValueError):
                invalid.validate()

    def test_fo_fork_schedule_and_validation(self):
        from hybrid.train import validate_fo_fork, validate_resume
        cfg = parse_args(["--mode", "hybrid", "--zo-norm-ratio", "0.3",
                          "--fo-checkpoint-dir", "/source", "--fo-checkpoint-step", "8000",
                          "--transition-steps", "1000"])
        resolved = recipe(cfg, 8)
        self.assertEqual(resolved, recipe(Config(), 8))
        self.assertEqual(transition_fraction(8000, cfg), 0)
        self.assertAlmostEqual(transition_fraction(8500, cfg), 500 / 999)
        self.assertEqual(transition_fraction(8999, cfg), 1)
        self.assertEqual(transition_fraction(9001, cfg), 1)
        self.assertEqual(transition_fraction(8000, replace(cfg, transition_steps=0)), 1)
        source = {"world_size": 8, "step": 8000, "user_config": asdict(Config()), "resolved": resolved}
        validate_fo_fork(cfg, source, 8, resolved)
        own = {"world_size": 8, "user_config": asdict(cfg)}
        validate_resume(replace(cfg, resume_step=8500), own, 8)
        with self.assertRaisesRegex(ValueError, "transition_steps"):
            validate_resume(replace(cfg, transition_steps=2000), own, 8)
        for bad in (replace(cfg, fo_checkpoint_step=7000), replace(cfg, seed=43)):
            with self.assertRaises(ValueError):
                validate_fo_fork(bad, source, 8, recipe(bad, 8))
        for bad in (replace(cfg, fo_checkpoint_dir=""), replace(cfg, transition_steps=1),
                    replace(cfg, mode="full_fo"), replace(cfg, zo_norm_ratio=None),
                    replace(cfg, transition_steps=20000)):
            with self.assertRaises(ValueError):
                recipe(bad, 8)
        old = asdict(Config())
        for key in ("fo_checkpoint_dir", "fo_checkpoint_step", "transition_steps"):
            del old[key]
        validate_resume(Config(), {"world_size": 8, "user_config": old}, 8)

    def test_default_schedule_is_linear(self):
        cfg = Config()
        self.assertEqual(lr_multiplier(0, 100, cfg), 1)
        self.assertAlmostEqual(lr_multiplier(50, 100, cfg), 0.55)
        self.assertAlmostEqual(lr_multiplier(100, 100, cfg), 0.1)

    def test_bad_batch_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            recipe(Config(device_batch_size=32), 8)

    def test_cli_overrides(self):
        cfg = parse_args(["--mode", "hybrid", "--zo-lambda", "0.5", "--no-compile",
                          "--prelude-mode", "joint_recompute", "--zo-input-epsilon", "0.002"])
        self.assertEqual(cfg.zo_epsilon, 1e-3)
        self.assertEqual(cfg.zo_lambda, 0.5)
        self.assertFalse(cfg.compile)
        self.assertEqual(cfg.prelude_mode, "joint_recompute")
        self.assertEqual(cfg.zo_input_epsilon, 0.002)
        self.assertEqual(cfg.zo_rho, 2)

    def test_rho_controls_input_scale_and_leaves_recipe_unchanged(self):
        cfg = parse_args(["--mode", "hybrid", "--prelude-mode", "joint", "--zo-rho", "0.1"])
        self.assertEqual(cfg.zo_rho, 0.1)
        self.assertAlmostEqual(cfg.zo_input_epsilon, 1e-4)
        self.assertEqual(recipe(cfg, 8), recipe(Config(), 8))
        self.assertAlmostEqual(replace(cfg, zo_epsilon=2e-3).zo_input_epsilon, 2e-4)
        self.assertEqual(execution_plan(cfg)["zo_rho"], 0.1)
        self.assertAlmostEqual(execution_plan(cfg)["zo_input_epsilon"], 1e-4)

    def test_legacy_input_epsilon_configuration(self):
        normalized = normalize_zo_settings({"zo_epsilon": 2e-3, "zo_input_epsilon": 1e-3})
        self.assertEqual(normalized, {"zo_epsilon": 2e-3, "zo_rho": 0.5})
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            normalize_zo_settings({"zo_input_epsilon": 1e-3, "zo_rho": 2})

    def test_launchers_pass_mode_and_preserve_arguments(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            # Capture argv only: this test never imports torch or starts workers.
            fake_python = Path(temp) / "capture python"
            fake_python.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                                   "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}))\n")
            fake_python.chmod(0o755)
            env = {**os.environ, "PYTHON_BIN": str(fake_python), "NPROC_PER_NODE": "8"}
            for mode in ("full_fo", "truncated", "hybrid"):
                result = subprocess.run(["bash", str(root / f"hybrid/scripts/{mode}.sh"),
                                         "--data-dir", "/tmp/data with spaces", "--stop-after", "100"],
                                        cwd=temp, env=env, check=True, text=True, capture_output=True)
                captured = json.loads(result.stdout)
                args = captured["argv"]
                self.assertEqual(captured["cwd"], str(root))
                self.assertEqual(args[args.index("--mode") + 1], mode)
                self.assertEqual(args[args.index("--data-dir") + 1], "/tmp/data with spaces")
                self.assertIn("--nproc_per_node=8", args)
            for prelude in PRELUDE_MODES:
                result = subprocess.run(["bash", str(root / f"hybrid/scripts/hybrid_{prelude}.sh"),
                                         "--stop-after", "100"], cwd=temp, env=env,
                                        check=True, text=True, capture_output=True)
                args = json.loads(result.stdout)["argv"]
                self.assertEqual(args[args.index("--mode") + 1], "hybrid")
                self.assertEqual(args[args.index("--prelude-mode") + 1], prelude)
            result = subprocess.run(["bash", str(root / "hybrid/scripts/run_prelude_modes.sh"),
                                     "--stop-after", "100"], cwd=temp, env=env,
                                    check=True, text=True, capture_output=True)
            launches = [json.loads(line)["argv"] for line in result.stdout.splitlines()]
            self.assertEqual([args[args.index("--prelude-mode") + 1] for args in launches], list(PRELUDE_MODES))


if __name__ == "__main__":
    unittest.main()
