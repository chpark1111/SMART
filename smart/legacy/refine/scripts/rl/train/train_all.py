import os
import signal
import subprocess

num_program = 10
processes = []
for i in range(num_program):
    process = subprocess.Popen(
        "./scripts/rl/train/train%d.sh" % (i),
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    processes.append(process)

for i, p in enumerate(processes):
    if p.wait() != 0:
        print("There was an error in train%d" % (i))

try:
    os.kill(-os.getpid(), signal.SIGINT)
except ProcessLookupError:
    print("Exited normally")
