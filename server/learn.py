import subprocess

arg = "kai"

subprocess.call("./learn.sh '%s'" % str(arg), shell=True)
