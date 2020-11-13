#!/usr/bin/env python3
import os
import time
import sys
import subprocess

import cereal.messaging as messaging
from common.basedir import BASEDIR
from common.params import Params
from selfdrive.test.helpers import set_params_enabled

def cputime_total(ct):
  return ct.cpuUser + ct.cpuSystem + ct.cpuChildrenUser + ct.cpuChildrenSystem


def print_cpu_usage(first_proc, last_proc):
  procs = [
    #("./_modeld", 7.12),
    ("./camerad", 7.07),
    ("./_dmonitoringmodeld", 3.5),
  ]

  r = True
  dt = (last_proc.logMonoTime - first_proc.logMonoTime) / 1e9
  result = "------------------------------------------------\n"
  for proc_name, normal_cpu_usage in procs:
    try:
      first = [p for p in first_proc.procLog.procs if proc_name in p.cmdline][0]
      last = [p for p in last_proc.procLog.procs if proc_name in p.cmdline][0]
      cpu_time = cputime_total(last) - cputime_total(first)
      cpu_usage = cpu_time / dt * 100.
      if cpu_usage > max(normal_cpu_usage * 1.1, normal_cpu_usage + 5.0):
        result += f"Warning {proc_name} using more CPU than normal\n"
        r = False
      elif cpu_usage < min(normal_cpu_usage * 0.65, max(normal_cpu_usage - 1.0, 0.0)):
        result += f"Warning {proc_name} using less CPU than normal\n"
        r = False
      result += f"{proc_name.ljust(35)}  {cpu_usage:.2f}%\n"
    except IndexError:
      result += f"{proc_name.ljust(35)}  NO METRICS FOUND\n"
      r = False
  result += "------------------------------------------------\n"
  print(result)
  return r

def test_cpu_usage():
  cpu_ok = False

  Params().delete("CarParams")

  # start manager
  manager_path = os.path.join(BASEDIR, "selfdrive/manager.py")
  manager_proc = subprocess.Popen(["python", manager_path])
  try:
    proc_sock = messaging.sub_sock('procLog', conflate=True, timeout=2000)

    # wait until everything's started
    start_time = time.monotonic()
    while time.monotonic() - start_time < 210:
      if Params().get("CarParams") is not None:
        break
      time.sleep(2)

    # take first sample
    time.sleep(5)
    first_proc = messaging.recv_sock(proc_sock, wait=True)
    if first_proc is None:
      raise Exception("\n\nTEST FAILED: progLog recv timed out\n\n")

    # run for a minute and get last sample
    time.sleep(20)
    last_proc = messaging.recv_sock(proc_sock, wait=True)
    cpu_ok = print_cpu_usage(first_proc, last_proc)
  finally:
    manager_proc.terminate()
    ret = manager_proc.wait(20)
    if ret is None:
      manager_proc.kill()
  return cpu_ok

if __name__ == "__main__":
  set_params_enabled()
  Params().delete("CarParams")

  import time

  for n in range(int(os.getenv("LOOP", "1"))):
    try:
      start_t = time.monotonic()
      passed = test_cpu_usage()
      if not passed:
        raise Exception
      print("\n\nPASSED RUN ", n, f", took {time.monotonic()-start_t}s\n\n")
    except Exception as e:
      print("\n\nFAILED ON RUN ", n)
      print(e)
      sys.exit(n)
