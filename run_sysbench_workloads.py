"""
/*
 * Software Name : Microtune
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the <license-name>,
 * see the "LICENSE.txt" file for more details or <license-url>
 *
 * <Authors: optional: see CONTRIBUTORS.md
 * Software description: MicroTune is a RL-based DBMS Buffer Pool Auto-Tuning for Optimal and Economical Memory Utilization. Consumed RAM is continously and optimally adjusted in conformance of a SLA constraint (maximum mean latency).
 */
"""

import time
from bandits.tools.sysbench.sbrequest import SysbenchRequest

#SysbenchRequest.set("host", "workload_sb_dev.local")
#SysbenchRequest.set("host", "workload_sb.local")
#SysbenchRequest.set("host", "192.168.0.136") #TUN04 (dedicated to workload generators)

# Admin request
#wl = SysbenchRequest(dbhost="192.168.0.206", id="oltp_read_write", tables=3, tablesize=50000) #TUN03
#wl = SysbenchRequest(dbhost="db.local", id="oltp_read_write", tables=2, tablesize=500)
#wl = SysbenchRequest(dbhost="192.168.0.212", id="oltp_read_write", tables=15, tablesize=50000) #TUN02
#wl = SysbenchRequest(httphost="workload_sb.local", dbhost="db.local", id="oltp_read_write", tables=35, tablesize=500000) #TUN06
w0 = SysbenchRequest(httphost="192.168.0.136", dbhost="192.168.0.143", id="oltp_read_write", tables=10, tablesize=1000) #TUN06
wl0 = SysbenchRequest(httphost="192.168.0.136", dbhost="192.168.0.143", id="oltp_read_write", tables=50, tablesize=1000000) #TUN06
wl1 = SysbenchRequest(httphost="192.168.0.136", dbhost="192.168.0.143", id="oltp_read_write", tables=10, tablesize=100000) #TUN06
wl2 = SysbenchRequest(httphost="192.168.0.136", dbhost="192.168.0.143", id="oltp_read_write", tables=15, tablesize=150000) #TUN06
wl3 = SysbenchRequest(httphost="192.168.0.136", dbhost="192.168.0.143", id="oltp_read_write", tables=20, tablesize=200000) #TUN06

# Load: Duration, VUS, Randtype
loads = [ 
#    { "w": wl1, "d": 120, "v": 3, "r": "uniform"},
#    { "w": wl1, "d": 240, "v": 7, "r": "special"},
#    { "w": wl2, "d": 120, "v": 7, "r": "uniform"},
#    { "w": wl1, "d": 60, "v": 2, "r": "special"},
#    { "w": wl3, "d": 120, "v": 5, "r": "uniform"},
#    { "w": wl1, "d": 240, "v": 5, "r": "special"},
#    { "w": wl1, "d": 480, "v": 1, "r": "gaussian"},
#    { "w": wl3, "d": 120, "v": 5, "r": "uniform"},
    { "w": wl3, "d": 600, "v": 8, "r": "uniform"},
    { "w": wl1, "d": 600, "v": 10, "r": "uniform"},
    { "w": wl1, "d": 600, "v": 12, "r": "uniform"},
    { "w": wl2, "d": 600, "v": 6, "r": "uniform"},
    { "w": wl2, "d": 600, "v": 10, "r": "uniform"},
    { "w": wl2, "d": 600, "v": 12, "r": "uniform"},
]

# For test
loads = [ { "w": w0, "d": 600, "v": 3, "r": "uniform"} ]

# A representative load test
wYif = SysbenchRequest(httphost="192.168.0.136", dbhost="192.168.0.143", id="oltp_read_write", tables=50, tablesize=1000000) #TUN06
loads = [ 
    { "w": wYif, "d": 2500, "v": 2, "r": "gaussian"},
    { "w": wYif, "d": 2500, "v": 8, "r": "gaussian"},
    { "w": wYif, "d": 2500, "v": 4, "r": "gaussian"},
    { "w": wYif, "d": 2500, "v": 3, "r": "uniform"},
    { "w": wYif, "d": 2500, "v": 2, "r": "special"},
    { "w": wYif, "d": 2500, "v": 2, "r": "special"},
    { "w": wYif, "d": 2500, "v": 1, "r": "special"},
    { "w": wYif, "d": 2500, "v": 1, "r": "special"},
]

prev_wl = None
wYif.cancel()
time.sleep(10)

for cl in loads:
    wl = cl["w"]
    if wl != prev_wl:
        prev_wl = wl
        print("Admin...")
        #wl.admin(cmd="cleanup", sleep_after=0)
        #wl.admin(cmd="prepare", sleep_after=1)
        print("Admin OK")

    print(f'ASync Load vus={cl["v"]} during {cl["d"]}s')
#    wl.load(vus=cl["v"], duration=cl["d"], randtype=cl["r"])
    wl.load(vus=cl["v"], duration=0, randtype=cl["r"])
    print(f'Wait {cl["d"]+3}s')
    time.sleep(cl["d"]+3)
    wl.cancel() # Just in case...

exit(0)