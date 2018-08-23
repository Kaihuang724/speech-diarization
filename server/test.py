import subprocess

def record_test():
    arg = "kai"
    subprocess.call("./test.sh '%s'" % str(arg), shell=True)
