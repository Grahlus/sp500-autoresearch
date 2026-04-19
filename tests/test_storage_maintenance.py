import tempfile
import unittest
from pathlib import Path

from storage_maintenance import rotate_logs


class StorageMaintenanceTests(unittest.TestCase):
    def test_rotate_logs_preserves_active_inode_and_archives_current_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            log_path = log_dir / "autonomous_research.log"
            original = b"line-1\nline-2\nline-3\n"
            log_path.write_bytes(original)
            before_inode = log_path.stat().st_ino

            report = rotate_logs(log_dir=log_dir, max_bytes=8, retain_count=2, dry_run=False)

            self.assertTrue(report["logs"][0]["rotated"])
            self.assertEqual(log_path.stat().st_ino, before_inode)
            self.assertEqual(log_path.read_bytes(), b"")
            archived = (log_dir / "autonomous_research.log.1").read_bytes()
            self.assertLessEqual(len(archived), 8)
            self.assertEqual(report["rotated_count"], 1)
            self.assertEqual(report["retained_count"], 2)

    def test_rotate_logs_shifts_existing_archives_and_caps_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            log_path = log_dir / "autonomous_research.log"
            log_path.write_text("active-a\n")
            (log_dir / "autonomous_research.log.1").write_text("archive-1\n")
            (log_dir / "autonomous_research.log.2").write_text("archive-2\n")
            (log_dir / "autonomous_research.log.3").write_text("archive-3\n")

            rotate_logs(log_dir=log_dir, max_bytes=1, retain_count=2, dry_run=False)

            self.assertLessEqual((log_dir / "autonomous_research.log.1").stat().st_size, 1)
            self.assertLessEqual((log_dir / "autonomous_research.log.2").stat().st_size, 1)
            self.assertFalse((log_dir / "autonomous_research.log.3").exists())

    def test_rotate_logs_clamps_zero_retain_count_to_one_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            log_path = log_dir / "autonomous_research.log"
            log_path.write_text("active-a\n")

            report = rotate_logs(log_dir=log_dir, max_bytes=1, retain_count=0, dry_run=False)

            self.assertEqual(report["retained_count"], 1)
            self.assertTrue((log_dir / "autonomous_research.log.1").exists())
            self.assertLessEqual((log_dir / "autonomous_research.log.1").stat().st_size, 1)

    def test_rotate_logs_trims_oversized_existing_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            log_path = log_dir / "autonomous_research.log"
            log_path.write_text("active\n")
            archive_path = log_dir / "autonomous_research.log.1"
            archive_path.write_text("line-1\nline-2\nline-3\nline-4\n")

            report = rotate_logs(log_dir=log_dir, max_bytes=12, retain_count=2, dry_run=False)

            self.assertEqual(report["trimmed_count"], 1)
            self.assertLessEqual(archive_path.stat().st_size, 12)


if __name__ == "__main__":
    unittest.main()
