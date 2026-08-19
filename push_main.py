import subprocess
import os

os.chdir(r"e:\AI_thucchien\lab\Day19-Track2-2A202601682-LuongBaoLong")

# Push to main directly
result = subprocess.run(
    ["git", "push", "-u", "origin", "main"],
    capture_output=True, text=True
)
print("Push result:", result.returncode)
print("stdout:", result.stdout)
print("stderr:", result.stderr)
