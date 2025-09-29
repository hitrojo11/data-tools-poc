# tests/test_snapshot_cleanup.py
import os
import tempfile
import time
from utils import cleanup_orphan_snapshots, SNAPSHOT_PREFIX, SNAPSHOT_SUFFIX


def test_cleanup_removes_old_snapshots(tmp_path):
    temp_dir = tempfile.gettempdir()
    # create two fake snapshots: one old, one recent
    old_name = SNAPSHOT_PREFIX + "old" + SNAPSHOT_SUFFIX
    recent_name = SNAPSHOT_PREFIX + "recent" + SNAPSHOT_SUFFIX
    old_path = os.path.join(temp_dir, old_name)
    recent_path = os.path.join(temp_dir, recent_name)
    with open(old_path, "wb") as f:
        f.write(b"old")
    with open(recent_path, "wb") as f:
        f.write(b"recent")
    # set old mtime to 2 days ago
    two_days = time.time() - (2 * 24 * 3600)
    os.utime(old_path, (two_days, two_days))

    removed = cleanup_orphan_snapshots(ttl_hours=24.0, temp_dir=temp_dir)
    assert removed >= 1
    # cleanup may remove both if older; ensure at least old is removed
    assert not os.path.exists(old_path)
    # remove the recent file as cleanup may or may not have removed it; ensure test leaves temp clean
    try:
        if os.path.exists(recent_path):
            os.remove(recent_path)
    except Exception:
        pass
