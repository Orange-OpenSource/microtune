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
import requests
import time

class SysbenchRequest():
    def __init__(self, httphost="workload_sb.local", httpport=8080, dbhost="db.local", dbport=3306, id="oltp_read_write", tables=3, tablesize=1000):
        self._dbhost = dbhost
        self._dbport = dbport
        self._adminUrl = "http://"+httphost+":"+str(httpport)+"/admin"
        self._loadUrl = "http://"+httphost+":"+str(httpport)+"/"
        self._cancelUrl = "http://"+httphost+":"+str(httpport)+"/cancel"
        self.id=id
        self.tables = tables
        self.tablesize=tablesize

    def getType(self) -> str:
        return self.id
    
    def getTablesSize(self) -> dict:
        return { "tables": self.tables, "table_rows": self.tablesize}
    
    def admin(self, cmd="cleanup", sleep_after=0):
        ploads = {
            'host': self._dbhost,
            'id':self.id,
            'tables': self.tables,
            'tablesize': self.tablesize,
            'cmd': cmd,
            }
            
        r = requests.get(self._adminUrl, params=ploads)
        #print(r.text)
        print("Request done: ", r.url)
        if sleep_after > 0:
            time.sleep(sleep_after)

    # Randtype:
    #  See https://severalnines.com/blog/how-benchmark-performance-mysql-mariadb-using-sysbench/
    #  Random numbers distribution {uniform, gaussian, special, pareto, zipfian} to use by default [special]
    def load(self, vus=1, duration=3, maxrequests=0, randtype=""):
        ploads = { 
            'host': self._dbhost,
            'id':self.id, 
            'vus':vus, 
            'duration':duration,
            'maxrequests': maxrequests,
            'tables': self.tables,
            'tablesize':self.tablesize,
            }
        if randtype != "":
            ploads["randtype"] = randtype

        print(self._loadUrl)
        print(ploads)
        r = requests.get(self._loadUrl,params=ploads)
        #print("REQ Done")
        print("Request done: ", r.url)
        #print(r.text)

    def cancel(self):
        ploads = { 
            }

        print(self._cancelUrl)
        print(ploads)
        r = requests.get(self._cancelUrl,params=ploads)
        #print("REQ Done")
        print("Request done: ", r.url)
