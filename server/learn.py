import subprocess

def record_learn():
    arg = "kai"

    subprocess.call("./learn.sh '%s'" % str(arg), shell=True)
