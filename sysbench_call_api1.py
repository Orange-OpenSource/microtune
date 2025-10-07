"""
/*
 * Software Name : Microtune
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * see the "LICENSE" file for more details
 *
 * Authors: see CONTRIBUTORS.md
 * Software description: MicroTune is a RL-based DBMS Buffer Pool Auto-Tuning for Optimal and Economical Memory Utilization. Consumed RAM is continously and optimally adjusted in conformance of a SLA constraint (maximum mean latency).
 */
"""

import time
from bandits.tools.sysbench.sbrequest import SysbenchRequest

wl1 = SysbenchRequest(httphost="workload_sb.local", dbhost="db.local", id="oltp_read_write", tables=30, tablesize=220000)
wl2 = SysbenchRequest(httphost="workload_sb.local", dbhost="db.local", id="oltp_read_write", tables=25, tablesize=110000)

# WorkLoad, Duration, VUS, Randtype
loads = [ 
    { "w": wl1, "d": 100, "v": 6, "r": "uniform"},
    { "w": wl2, "d": 200, "v": 6, "r": "uniform"},
    { "w": wl1, "d": 200, "v": 3, "r": "uniform"},
]

prev_wl = None
wl1.cancel()
#exit(0) # Cancel any previous load
#time.sleep(1)

for cl in loads:
    wl = cl["w"]
    if wl != prev_wl:
        prev_wl = wl
        print("Admin...")
        wl.admin(cmd="cleanup", sleep_after=0)
        wl.admin(cmd="prepare", sleep_after=1)
        print("Admin OK")

    print(f'ASync Load vus={cl["v"]} during {cl["d"]}s')
#    wl.load(vus=cl["v"], duration=cl["d"], randtype=cl["r"])
    wl.load(vus=cl["v"], duration=0, randtype=cl["r"])
    print(f'Wait {cl["d"]+3}s')
    time.sleep(cl["d"]+3)
    wl.cancel() # In case where duration passed to wl.load() is 0...

exit(0)
