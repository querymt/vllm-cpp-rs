"""Deterministic tests of release recipes; no native build or model required."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "path+file:///fixture/vllm-cpp#0.0.2"


class ReleaseTooling(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vllm-release-tooling-")
        self.work = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def recipe(self, *args, env=None):
        return subprocess.run(
            ["just", "--justfile", str(ROOT / "Justfile"), *map(str, args)],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=15,
        )

    def artifact(self, executable, package=PACKAGE, target="qwen3", kind="test"):
        return {
            "reason": "compiler-artifact", "package_id": package,
            "target": {"name": target, "kind": [kind]},
            "profile": {"test": True}, "executable": str(executable),
        }

    def select(self, messages):
        path = self.work / "messages.json"
        path.write_text("".join(json.dumps(message) + "\n" for message in messages))
        return self.recipe("_test-binary", path, PACKAGE, "qwen3", "test")

    def test_artifact_selection_ignores_newer_stale_and_foreign_executables(self):
        current = self.work / "qwen3-current"
        stale = self.work / "qwen3-stale"
        for path in [current, stale]:
            path.touch(mode=0o755)
        os.utime(current, (1, 1))
        os.utime(stale, (999999, 999999))
        result = self.select([
            self.artifact(stale, package="registry+other#foreign@0.0.2"),
            self.artifact(stale, kind="example"),
            self.artifact(stale, target="safe_api"),
            self.artifact(current),
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(current))

    def test_artifact_selection_rejects_missing_or_ambiguous_matches(self):
        for messages in [[], [self.artifact("one"), self.artifact("two")]]:
            with self.subTest(messages=messages):
                self.assertNotEqual(self.select(messages).returncode, 0)

    def assert_results(self, text, focused=False, name="suite"):
        path = self.work / "output.txt"
        path.write_text(text)
        return self.recipe("_assert-rust-tests", path, name, str(focused).lower())

    def test_complete_suite_requires_nonzero_matching_counts(self):
        self.assertEqual(self.assert_results(
            "running 3 tests\ntest result: ok. 3 passed; 0 failed; 0 ignored;\n"
        ).returncode, 0)
        for text in [
            "running 0 tests\ntest result: ok. 0 passed; 0 failed;\n",
            "running 3 tests\ntest result: ok. 1 passed; 0 failed; 2 ignored;\n",
            "running 1 test\ntest result: FAILED. 0 passed; 1 failed;\n",
            "running 1 test\nrunning 1 test\ntest result: ok. 1 passed; 0 failed;\n",
        ]:
            with self.subTest(text=text):
                self.assertNotEqual(self.assert_results(text).returncode, 0)

    def test_focused_test_requires_exactly_one_named_pass(self):
        text = "running 1 test\ntest lifecycle ... ok\ntest result: ok. 1 passed; 0 failed;\n"
        self.assertEqual(self.assert_results(text, True, "lifecycle").returncode, 0)
        native_output = text.replace("... ok", "... native diagnostic\nok")
        self.assertEqual(self.assert_results(native_output, True, "lifecycle").returncode, 0)
        for output in [
            "running 0 tests\ntest result: ok. 0 passed; 0 failed;\n",
            text.replace("lifecycle", "different"),
            text.replace("1 passed", "0 passed"),
        ]:
            with self.subTest(output=output):
                self.assertNotEqual(self.assert_results(output, True, "lifecycle").returncode, 0)

    def test_optional_lanes_use_cargo_artifacts_and_result_checks(self):
        text = (ROOT / "Justfile").read_text()
        for start, end in [("sanitizers model=", "# Run selected request"),
                           ("tsan model=", "# Check Just")]:
            recipe = text.split(start, 1)[1].split(end, 1)[0]
            self.assertIn("--message-format=json", recipe)
            self.assertIn("_test-binary", recipe)
            self.assertIn("_assert-rust-tests", recipe)
            self.assertNotIn("find ", recipe)

    def test_package_payload_ignores_directory_metadata_and_links(self):
        for count in [0, 100]:
            archive = self.work / f"payload-{count}.crate"
            with tarfile.open(archive, "w:gz") as stream:
                for number in range(count):
                    member = tarfile.TarInfo(f"pkg/empty-{number}/")
                    member.type = tarfile.DIRTYPE
                    stream.addfile(member)
                link = tarfile.TarInfo("pkg/link")
                link.type = tarfile.SYMTYPE
                link.linkname = "payload"
                stream.addfile(link)
                member = tarfile.TarInfo("pkg/payload")
                member.size = 53
                stream.addfile(member, io.BytesIO(b"x" * 53))
            result = self.recipe("_package-payload", archive)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "53")

    def test_package_metrics_are_diagnostics_not_release_caps(self):
        package = (ROOT / "Justfile").read_text().split(
            "\npackage-test:\n", 1
        )[1].split("\n# Build and run the focused native", 1)[0]
        for crate in ["sys", "safe"]:
            for suffix in ["package_size", "payload_size", "entry_count", "package_sha256"]:
                metric = f"{crate}_{suffix}"
                uses = [line.strip() for line in package.splitlines() if metric in line]
                with self.subTest(metric=metric):
                    # Metrics may be computed and printed, never used as gate conditions.
                    self.assertEqual(len(uses), 2, uses)
                    self.assertTrue(uses[0].startswith(f"{metric}=$("), uses)
                    self.assertIn(f'"${metric}"', uses[1])
            self.assertIn(f"{crate} package: %d entries, %d payload bytes, "
                          "%d compressed bytes, sha256 %s", package)
            self.assertIn(f'_package-payload "${crate}_package"', package)

    def test_package_content_and_compatibility_checks_remain_required(self):
        package = (ROOT / "Justfile").read_text().split(
            "\npackage-test:\n", 1
        )[1].split("\n# Build and run the focused native", 1)[0]
        for check in [
            'diff -u "$sys_list" <(archive_inventory',
            'diff -u "$safe_list" <(archive_inventory',
            'diff -u "$sys_expected" "$sys_list"',
            'diff -u "$safe_expected" "$safe_list"',
            'required_sys_members=(', 'done < "$safe_expected"',
            'denied_sys_members=(', '"$forbidden_pattern" "$listing"',
            'scan_authored_paths "$sys_root"', 'scan_authored_paths "$safe_root"',
            '"$sys_license_actual"', 'licenses/DOCTEST-MIT.txt',
            'check_package_links "$sys_root" README.md THIRD_PARTY.md',
            'check_package_links "$safe_root" README.md examples/README.md',
            'cargo metadata --manifest-path "$sys_root/Cargo.toml"',
            'cargo metadata --manifest-path "$safe_root/Cargo.toml"',
            '.req == ("=" + $version)',
            '"$sys_root" "$temp/extracted-link-modes"',
            'cargo test --locked --release --offline --features bundled,serde',
            'cd "$sys_consumer"', 'cd "$safe_consumer"',
            '.manifest_path == $safe', '.manifest_path == $sys',
        ]:
            with self.subTest(check=check):
                self.assertIn(check, package)
        self.assertEqual(package.count('cargo run --locked --release --offline'), 2)
        self.assertIn("package-test publish-dry-run",
                      (ROOT / "Justfile").read_text().split("\nci:", 1)[1])

    def run_native(self, status):
        tools = self.work / "bin"
        tools.mkdir()
        cmake = tools / "cmake"
        cmake.write_text("#!/usr/bin/env bash\nexit 0\n")
        cmake.chmod(0o755)
        ctest = tools / "ctest"
        ctest.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ["MOCK_LOG"], "a") as log:
    log.write(json.dumps({"args": sys.argv[1:], "sched": os.getenv("VT_ASYNC_SCHED"), "runner": os.getenv("VT_ASYNC_RUNNER")}) + "\\n")
if "-N" in sys.argv:
    print("  Test #1: test_capi\\nTotal Tests: 1")
    sys.exit(0)
if os.getenv("VT_ASYNC_SCHED") == "0" or os.environ["MOCK_STATUS"] == "0":
    print("100% tests passed, 0 tests failed out of 1")
    sys.exit(0)
print("""TEST CASE: capi: vllm_complete_stream early-stop tears the request down cleanly
test_capi.cpp:680: ERROR: CHECK( acc.deltas == 2 ) is NOT correct!
  values: CHECK( 1 == 2 )
[doctest] test cases: 49 | 48 passed | 1 failed | 0 skipped
[doctest] assertions: 100 | 99 passed | 1 failed |
0% tests passed, 1 tests failed out of 1
  1 - test_capi (Failed)""")
sys.exit(int(os.environ["MOCK_STATUS"]))
''')
        ctest.chmod(0o755)
        log = self.work / "ctest.jsonl"
        env = dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}",
                   CARGO_TARGET_DIR=str(self.work / "target"), MOCK_LOG=str(log),
                   MOCK_STATUS=str(status), VT_ASYNC_SCHED="inherited", VT_ASYNC_RUNNER="inherited")
        result = self.recipe("native-capi", env=env)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(calls), 2, result.stdout + result.stderr)
        self.assertIsNone(calls[1]["sched"])
        self.assertIsNone(calls[1]["runner"])
        return result

    def test_native_success_uses_default_environment(self):
        result = self.run_native(0)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_failure_stays_failed_without_alternate_configuration(self):
        result = self.run_native(8)
        self.assertEqual(result.returncode, 8, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
