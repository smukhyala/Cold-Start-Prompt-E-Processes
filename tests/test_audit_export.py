#!/usr/bin/env python3
"""Tests for audit export matrices."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import pandas as pd

from experiments.growing_bandits.audit import export_matrices


class TestExportMatrices:
    """Test pool export functionality."""

    @pytest.fixture
    def out_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def load_exported_csv(self, out_dir: Path, pool_name: str) -> pd.DataFrame:
        """Load an exported CSV file."""
        path = out_dir / f"{pool_name}.csv"
        if not path.exists():
            pytest.skip(f"Data file not found: {path}")
        return pd.read_csv(path)

    def test_export_G(self, out_dir: Path):
        """Test G pool export."""
        export_matrices.export_pool("G", out_dir)
        df = self.load_exported_csv(out_dir, "G")

        assert df["arm_id"].nunique() == 50
        assert df["task_id"].nunique() == 30
        assert set(df["success"].unique()).issubset({0, 1})

        # Check no duplicates
        duplicates = df.groupby(["arm_id", "task_id", "replicate"]).size()
        assert all(duplicates == 1)

    def test_export_F(self, out_dir: Path):
        """Test F pool export."""
        export_matrices.export_pool("F", out_dir)
        df = self.load_exported_csv(out_dir, "F")

        assert df["arm_id"].nunique() == 50
        assert df["task_id"].nunique() == 30
        assert set(df["success"].unique()).issubset({0, 1})

        # Check no duplicates
        duplicates = df.groupby(["arm_id", "task_id", "replicate"]).size()
        assert all(duplicates == 1)

    def test_export_GLK30(self, out_dir: Path):
        """Test GLK30 pool export."""
        export_matrices.export_pool("GLK30", out_dir)
        df = self.load_exported_csv(out_dir, "GLK30")

        assert df["arm_id"].nunique() == 40
        assert df["task_id"].nunique() == 30
        assert set(df["success"].unique()).issubset({0, 1})

        # Check no duplicates
        duplicates = df.groupby(["arm_id", "task_id", "replicate"]).size()
        assert all(duplicates == 1)

    def test_export_GLK(self, out_dir: Path):
        """Test GLK pool export."""
        export_matrices.export_pool("GLK", out_dir)
        df = self.load_exported_csv(out_dir, "GLK")

        assert df["arm_id"].nunique() == 40
        assert df["task_id"].nunique() == 30
        assert set(df["success"].unique()).issubset({0, 1})

        # Check no duplicates
        duplicates = df.groupby(["arm_id", "task_id", "replicate"]).size()
        assert all(duplicates == 1)

    def test_export_gitlab18(self, out_dir: Path):
        """Test GitLab 18-arm pool export."""
        export_matrices.export_pool("gitlab18", out_dir)
        df = self.load_exported_csv(out_dir, "gitlab18")

        assert df["arm_id"].nunique() == 18
        assert df["task_id"].nunique() == 40
        assert set(df["success"].unique()).issubset({0, 1})

        # Check no duplicates
        duplicates = df.groupby(["arm_id", "task_id", "replicate"]).size()
        assert all(duplicates == 1)

    def test_export_gmail_paired12(self, out_dir: Path):
        """Test Gmail paired 12-arm pool export."""
        export_matrices.export_pool("gmail_paired12", out_dir)
        df = self.load_exported_csv(out_dir, "gmail_paired12")

        assert df["arm_id"].nunique() == 12
        assert df["task_id"].nunique() == 60
        assert set(df["success"].unique()).issubset({0, 1})

        # Check no duplicates
        duplicates = df.groupby(["arm_id", "task_id", "replicate"]).size()
        assert all(duplicates == 1)

    def test_export_shopping_paired12(self, out_dir: Path):
        """Test Shopping paired 12-arm pool export."""
        export_matrices.export_pool("shopping_paired12", out_dir)
        df = self.load_exported_csv(out_dir, "shopping_paired12")

        assert df["arm_id"].nunique() == 12
        assert df["task_id"].nunique() == 80
        assert set(df["success"].unique()).issubset({0, 1})

        # Check no duplicates
        duplicates = df.groupby(["arm_id", "task_id", "replicate"]).size()
        assert all(duplicates == 1)

    def test_sanity_check_success_rates(self, out_dir: Path):
        """Verify sanity-check success rates match expectations."""
        # Gmail paired
        export_matrices.export_pool("gmail_paired12", out_dir)
        df = pd.read_csv(out_dir / "gmail_paired12.csv")

        junior_reactive_success = df[df["arm_id"] == "junior_reactive"]["success"].sum()
        assert junior_reactive_success == 42, f"Expected 42, got {junior_reactive_success}"

        planner_success = df[df["arm_id"] == "planner"]["success"].sum()
        assert planner_success == 27, f"Expected 27, got {planner_success}"

        # Shopping paired
        export_matrices.export_pool("shopping_paired12", out_dir)
        df = pd.read_csv(out_dir / "shopping_paired12.csv")

        rapid_success = df[df["arm_id"] == "rapid"]["success"].sum()
        assert rapid_success == 37, f"Expected 37, got {rapid_success}"

        baseline_success = df[df["arm_id"] == "baseline"]["success"].sum()
        assert baseline_success == 37, f"Expected 37, got {baseline_success}"

        # GitLab
        export_matrices.export_pool("gitlab18", out_dir)
        df = pd.read_csv(out_dir / "gitlab18.csv")

        oracle_success = df[df["arm_id"] == "gitlab_oracle_operator"]["success"].sum()
        assert oracle_success == 35, f"Expected 35, got {oracle_success}"

    def test_csv_schema(self, out_dir: Path):
        """Verify all exported CSVs have correct schema."""
        export_matrices.export_pool("G", out_dir)
        df = pd.read_csv(out_dir / "G.csv")

        expected_columns = {"pool", "arm_id", "task_id", "replicate", "success", "steps", "cost_usd"}
        assert set(df.columns) == expected_columns, f"Unexpected columns: {set(df.columns)}"

    def test_summary_csv_created(self, out_dir: Path):
        """Verify summary.csv is created."""
        export_matrices.export_all(out_dir)

        summary_path = out_dir / "summary.csv"
        assert summary_path.exists(), "summary.csv not created"

        df = pd.read_csv(summary_path)
        expected_columns = {
            "pool",
            "n_rows",
            "n_arms",
            "n_tasks",
            "n_primary_rows",
            "n_replicate_rows",
            "mean_success_primary",
            "n_dropped",
            "source",
        }
        assert set(df.columns) == expected_columns

    def test_all_pools_have_positive_rows(self, out_dir: Path):
        """Verify all pools have data."""
        export_matrices.export_all(out_dir)

        summary_path = out_dir / "summary.csv"
        df = pd.read_csv(summary_path)

        assert len(df) > 0, "No pools exported"
        assert all(df["n_rows"] > 0), "Some pools have no rows"
