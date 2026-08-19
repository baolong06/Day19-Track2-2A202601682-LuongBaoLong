import subprocess
import os

os.chdir(r"e:\AI_thucchien\lab\Day19-Track2-2A202601682-LuongBaoLong")

# Commit
result = subprocess.run(
    ["git", "commit", "-m", "Day19 Lab submission - LuongBaoLong"],
    capture_output=True, text=True
)
print("Commit result:", result.returncode)
print("stdout:", result.stdout)
print("stderr:", result.stderr)
