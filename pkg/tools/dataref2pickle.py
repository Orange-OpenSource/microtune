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

# Connect to Mongo DataRef Database and prepare then export all observation samples to a pickle file loadable by Microtune
# Usage example: python pkg/tools/dataref2pickle.py --version 14 --mongohost dataref-service --mongoport 27017 --dbname adbms-obs-ref-mariadb-11_1_3 --objective_margin 0.3

import sys
sys.path.append('.')
import argparse

from pkg.datasource.dataframes.obs_samples_dataframes import ObsSamplesDF 

def export(version="14",  mongohost="localhost", mongoport=27017, dbname="adbms-obs-ref-mariadb-11_1_3", objective_margin=0.3):
   picklefiles="./workloads"  
   obsdf = ObsSamplesDF(version=version)
   df = obsdf.getSimuData(mongohost=mongohost, mongoport=mongoport, dbname=dbname, objective_margin=objective_margin)
   output_filename =f'{picklefiles}_full_{version}'
   obsdf.saveToPickle(output_filename, df)
   print(f"Exported data to {output_filename}.pikcle") 

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Prepare and Export observation samples from MongoDB (DataRef Obs Samples) to a pickle file loadable by Microtune')
   parser.add_argument('--version', type=str, default="14", help='Version of the DBMS (default: 14)')
   parser.add_argument('--mongohost', type=str, default="dataref-service", help='MongoDB host (default: dataref-service)')
   parser.add_argument('--mongoport', type=int, default=27017, help='MongoDB port (default: 27017)')
   parser.add_argument('--dbname', type=str, default="adbms-obs-ref-mariadb-11_1_3", help='MongoDB database name (default: adbms-obs-ref-mariadb-11_1_3)')
   parser.add_argument('--objective_margin', type=float, default=0.3, help='Objective margin defining the "SLA OK" zone in which the STAY ARM is the good one (default: 0.3)')
   args = parser.parse_args()

   print(f"Exporting observation samples from MongoDB {args.dbname} (host={args.mongohost}, port={args.mongoport}) to a pickle file loadable by Microtune")
   export(version=args.version, mongohost=args.mongohost, mongoport=args.mongoport, dbname=args.dbname, objective_margin=args.objective_margin) 

