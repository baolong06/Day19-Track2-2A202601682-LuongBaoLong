import subprocess
import os

os.chdir(r"e:\AI_thucchien\lab\Day19-Track2-2A202601682-LuongBaoLong")

# Create new branch and push
result = subprocess.run(
    ["git", "push", "-u", "origin", "HEAD:day19-submission"],
    capture_output=True, text=True
)
print("Push result:", result.returncode)
print("stdout:", result.stdout)
print("stderr:", result.stderr)
