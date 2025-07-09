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


import numpy as np
import mysql.connector
from mysql.connector import errorcode
import time
from pandas import DataFrame
from datetime import datetime

from bandits.datasource.db.dbcon import DBCon
import bandits.datasource.db.dberrors as dberrors

# Connections: The number of connection attempts (successful or not) to the MySQL server, collected at the last observation time
# Threads_created: The number of threads created to handle connections
# threads_cache_hit: Cache miss rate on connections defined by Threads_created/Connections
# Threads_running: The number of threads that are not sleeping
# created_tmp_disk_tables: The number of internal on-disk temporary tables created by the server while executing statements.
# Binlog_cache_disk_use: The number of transactions that used the temporary binary log cache but that exceeded the value of binlog_cache_size and used a temporary file to store statements from the transaction
# Binlog_stmt_cache_disk_use: The number of nontransaction statements that used the binary log statement cache but that exceeded the value of binlog_stmt_cache_size and used a temporary file to store those statements
# LATER? Global_connection_memory: The memory used by all user connections to the server. Memory used by system threads or by the MySQL root account is included in the total, but such threads or users are not subject to disconnection due to memory usage. This memory is not calculated unless global_connection_memory_tracking is enabled (disabled by default). The Performance Schema must also be enabled.

""" 
# Performance_schema is NOT Information_schema. Performance_schema in MariaDB is deactivated by default for performance reasons.
# From https://github.com/ottertune/ot-agent/blob/main/driver/collector/mysql_collector.py
    # convert the time unit from ps to ms by dividing 1,000,000,000
    METRICS_LATENCY_HIST_SQL = (
        "SELECT bucket_number, bucket_timer_low / 1000000000 as bucket_timer_low, "
        "bucket_timer_high / 1000000000 as bucket_timer_high, count_bucket, "
        "count_bucket_and_lower, bucket_quantile FROM "
        "performance_schema.events_statements_histogram_global;"
    )
# See also: Metrics of type value and not counter: see https://github.com/HustAIsGroup/CDBTune/blob/master/environment/utils.py#L40
 """
class DBAdminMySql():
    # See args list here: https://dev.mysql.com/doc/connector-python/en/connector-python-connectargs.html
    config = {
        "host": "db.local", 
        "port": 3306,  # Default MySQL port
        "user": "adbms",  # Since Granted rights have evolved, we can use adbms user instead of root
        "password": "adbms", 
        "database": "adbms",
        "raise_on_warnings": True,
    }

    #   - Global status variable is an array of the names of all DB Global Status variables
    # Questions: The number of statements executed by the server. This includes only statements sent to the server by clients and not stateme
    # Bytes_sent: The number of bytes sent to all clients.
    # Bytes_received: The number of bytes received from all clients.
    # buffer_pool_size_increment is 128 MB as default value 
    def __init__(self, servername: str = "mysql", serverversion: str = "0", dbhost="db.local", dbport=3306, user="adbms", password="adbms", database="adbms",
                 dynamicKnobsToDrive=[], buffer_pool_size_increment=134217728, other_knobs=[], 
                 global_status=[], information_schemas=[], information_schemas_mapping: dict = None,
                 sanity_statements=["LOGS"]):   #, env_metadata={}):
        self._servername = servername
        self._serverversion = serverversion
        self.config["host"] = dbhost
        self.config["port"] = dbport
        self.config["user"] = user
        self.config["password"] = password
        self.config["database"] = database
        self._sanity_statements = sanity_statements
        self._serverversion = serverversion
        self._dbCon = None

        self._metadata = {
            # System Variables (not read like Status variable !)
           "other_knobs": other_knobs,
            # See: https://dev.mysql.com/doc/refman/8.0/en/server-status-variables.html
           "global_status_variables": global_status, 
           "information_schemas": information_schemas,
           # Usage list is observed and collected in a more accurate way
           "usage_status_variables": [ "Questions", "Bytes_sent", "created_tmp_disk_tables" ]
        } 
        self._dynKnobsPrevious = {}
        self._dynKnobsUpdatedInDB = {}
        self._dynKnobs = {}
        for knob in dynamicKnobsToDrive:
           self._dynKnobs[knob] = "" 
        self._otherKnobs = {}
        self._dbStatus = { "__globvars_collected": False, "__complete_globavars_collected": False }
        self._ps_histo_global1 = None
        self._ps_histo_global2 = None
        self._min_buffer_pool_size = buffer_pool_size_increment # Default assignment 
        self._information_schemas= {} # Dictionary of collected Information_Schema 
        self._information_schemas_mapping = information_schemas_mapping

    # Get all enabled metric value of the information schema metrics (as specified at the object initialization) 
    # The doc https://mariadb.com/kb/en/information-schema-innodb_metrics-table/ is wrong. Unlike MySQL the STATUS field is replaced by ENABLED field in MariaDB :-(
    def _getInformationSchema(self):
        for schema in self._metadata["information_schemas"]:
            res = {}
            match schema:
                case "innodb_metrics":
                    values = self._dbCon.fetchAll("SELECT NAME, COUNT from information_schema."+schema+" where enabled=1")
                    for row in values:
                        res[row[0]] = row[1]
                case _:
                    values = self._dbCon.fetchAll("SELECT * from information_schema."+schema, dictionary=True)
                    res = {} if values is None or len(values)==0 else values[0]
            self._information_schemas[schema] = res

        #print(datetime.now(), "ALL SCHEMAAAAAAAAAA:", self._information_schemas)        

    #def _applyInformationSchemaMappingToStatus(self):
    #    if self._information_schemas_mapping:
    #        for key, val in  self._information_schemas_mapping  


    # Raises:
    # StatusVarGetError, KnobMetadataValueNotFound
    def _initGlobalStatus(self, otherKnobNames=[]):
        # Collect Information Schemas
        self._getInformationSchema()
            
        self._dbStatus = { "__globvars_collected": False, "__complete_globavars_collected": False }
        self._dbGetStatusAtStep(step="init", statusVars=self._metadata["global_status_variables"], scope="GLOBAL")
        self._dbGetStatusAtStep(step="init", statusVars=self._metadata["usage_status_variables"], scope="GLOBAL")

        # Pick up all other knobs values from DB server's GlOBAL parameters
        [ self._metadata["other_knobs"].append(k) for k in otherKnobNames if k not in self._metadata["other_knobs"] ] # Append to list of otherKnobs without duplicates
        for knob in self._metadata["other_knobs"]:
            try:
                self._otherKnobs[knob] = "NA"
                self._otherKnobs[knob] = self._dbGetGlobalVariable(knob)
            except dberrors.GlobalVarError as err:    
                raise dberrors.KnobMetadataValueNotFound(knob, "Metadata Other knob value not found")

        # Defines the minimal value of Buffer pool size from Chunk size and Pool instances, else keep the default value assegned with the increment value
        if self._otherKnobs.get("innodb_buffer_pool_chunk_size") is not None and self._otherKnobs.get("innodb_buffer_pool_instances") is not None: 
            self._min_buffer_pool_size = int(self._otherKnobs["innodb_buffer_pool_chunk_size"])*int(self._otherKnobs["innodb_buffer_pool_instances"])

    def getMinBufferPoolSize(self) -> int:
        return self._min_buffer_pool_size

    def isConnected(self) -> bool :
        if self._dbCon is None:
            return False
        return self._dbCon.isConnected()

    # Sanity check on the database, by executing a flush statements
    # Flush the choosen sanity statment(s) with "NO_WRITE_TO_BINLOG" to avoid writing to the binary log
    # The statements are defined here: https://dev.mysql.com/doc/refman/8.0/en/flush.html 
    # For MariaDB: https://mariadb.com/docs/server/reference/sql-statements/administrative-sql-statements/flush-commands/flush
    # If a statement is "LOGS" or "BINARY LOGS", it will fpurge BINARY LOGS before NOW().
    def sanity(self, duration=0.):
        for stmt in self._sanity_statements:
            try:
                self._dbCon.flushStatement(stmt)
                if stmt == "LOGS" or stmt == "BINARY LOGS":
                    req = 'PURGE BINARY LOGS BEFORE NOW();'
                    self._dbCon.statement(req)
            except mysql.connector.Error as err:
                print("Sanity statement failed:", stmt, "Error:", err)
        if duration > 0.:
            time.sleep(duration)

    def connect(self):
        """
        Connect to the database using the configuration provided.
        If the connection fails, it will raise an exception.
        Raise: DBConnexionError, DBServerVersionError
        """
        self._dbCon = DBCon(self.config)
        vers = self._dbCon.version()
        if vers["version"].startswith(self._serverversion) is False:
            raise dberrors.DBServerVersionError(vers["version"], self._serverversion)


    
    # Try to use the database, by executing a simple query. 
    # Check database readiness every 3 seconds, until the database exists and at least "n_tables" table exists and wait indefinitely...
    # Return the tables count present in the database if tables count >= "n_tables".
    def getDatabaseReady(self, n_tables: int=0) -> int:
        """
        Wait for the database to be ready, by executing a simple query.
        If the database is not ready, it will raise an exception.
        """
        count = -1
        dbname = self.config["database"]
        duration = 3

        # Wait for the database to be ready
        while count < n_tables:
            try:
                self._dbCon.statement("USE "+dbname+";")
                qres = self._dbCon.fetchOne("SELECT COUNT(*) as C FROM `information_schema`.`tables` WHERE `table_schema` = '"+dbname+"';")
                count = int(qres["C"])
            except mysql.connector.Error as err:
                print(dberrors.DBNotReadyError(dbname, err.msg))
            
            if count < n_tables:
                print(datetime.now(), "Waiting for the database to be ready... Tables count:", count, "Expected:", n_tables)
                time.sleep(duration)
                duration = min(60, duration+3)  # Increase the duration up to 60 seconds

        self._initGlobalStatus()
                
        return count
        
    def close(self):
        self._dbCon.close()

    def getMetaData(self):
        return self._metadata.copy()

    def getKnobsToDrive(self, sync_from_db=False):
        if sync_from_db:
            self._dbGetKnobsToDrive()
        return self._dynKnobs.copy()

    def getOtherKnobs(self):
        return self._otherKnobs.copy()

    def _collectUsageStatusValues(self, complete: bool = False):
        self._dbStatus["__status_collect_obs_duration"] = 0

        multi_observation_time = 1
        pcount=3
        qc=np.zeros(pcount)
        bc=np.zeros(pcount)
        ctdt=np.zeros(pcount)

        for m in range(0,pcount):
            self._dbGetStatusAtStep(step="start", statusVars=self._metadata["usage_status_variables"], scope="GLOBAL")
            time.sleep(multi_observation_time)
            self._dbGetStatusAtStep(step="end", statusVars=self._metadata["usage_status_variables"], scope="GLOBAL")

            qc[m] = int(self._dbStatus["Questions_diff"]) - 2
            bc[m] = int(self._dbStatus["Bytes_sent_diff"])
            ctdt[m] = int(self._dbStatus["created_tmp_disk_tables_diff"])
            time.sleep(0.33)
            self._dbStatus["__status_collect_obs_duration"] += multi_observation_time+0.33

        self._dbStatus["qps"] = np.mean(qc)//multi_observation_time
        bpsmean = np.mean(bc)
        self._dbStatus["Bps"] = np.mean(bc)//multi_observation_time
        self._dbStatus["KBps"] = (bpsmean/1024)//multi_observation_time
        self._dbStatus["created_tmp_disk_tables_per_sec"] = np.mean(ctdt)/multi_observation_time
        #self._dbStatus["threads_connected"] = int(self._dbStatus["threads_connected_end"])

        # Complete information on DB schema, Index size... ?
        # SELECT COUNT(*) FROM `information_schema`.`tables` WHERE `table_schema` = 'my_database_name';
        # SELECT SUM(TABLE_ROWS) FROM INFORMATION_SCHEMA.TABLES  WHERE TABLE_SCHEMA = '{your_db}';
        if self._dbStatus["__complete_globavars_collected"] is False and complete:
            # Get the database size and index size
            dbname = self.config["database"]
            req = 'SELECT table_schema "Schema", engine "Engine", ROUND(SUM(data_length)/1024/1024,2) "TableSizeMB", ROUND(SUM(index_length)/1024/1024,2) "IndexSizeMB" \
            FROM information_schema.tables \
            WHERE table_schema IN ("'+dbname+'") \
            AND engine IS NOT NULL \
            GROUP BY table_schema, engine;'
            qres = self._dbCon.fetchOne(req)
            if qres is None or len(qres) == 0:
                qres = { "Schema": dbname, "Engine": "NA", "TableSizeMB": 0, "IndexSizeMB": 0 }
            self._dbStatus["db_size"] = qres
            qres["TableSizeMB"] = round(float(qres["TableSizeMB"]), 2)
            qres["IndexSizeMB"] = round(float(qres["IndexSizeMB"]), 2)

            # Get the table count for the database
            qres = self._dbCon.fetchOne("SELECT COUNT(*) as count FROM `information_schema`.`tables` WHERE `table_schema` = '"+dbname+"';")
            if qres is None or len(qres) == 0:
                self._dbStatus["table_count"] = 0
            else:
                self._dbStatus["table_count"] =  int(qres["count"])

            # Get the row count for the database
            qres = self._dbCon.fetchOne("SELECT SUM(TABLE_ROWS) as count FROM INFORMATION_SCHEMA.TABLES  WHERE TABLE_SCHEMA = '"+dbname+"';")    
            if qres is None or qres["count"] is None or len(qres) == 0:
                self._dbStatus["count"] = 0
            else:
                print("ROW COUNT:", qres)
                self._dbStatus["row_count"] = int(qres["count"])
                self._dbStatus["__complete_globavars_collected"] = True


    # Get global status variables. Metadata must be defined as expected.
    # Some variables ae "instant" variables, we can pick up them for their instant value.
    # Some others (like max_connections_errors_max), can be took either instantly or we need a measurement on a lapse of time.
    # The lapse of time is defined by the "observation_time" passed as parameter. 
    # The variable "<glob_var_name>_begin" is the value just before the observation time.
    # The variable "<glob_var_name>_end" is the value just after the observation time.
    # The variable "<glob_var_name>_diff" provide the count (changes) which is equal to (<glob_var_name>_begin - <glob_var_name>_end)
    # - Flush DB status global variables
    # - Pick up Status values listed in the metadata["global_status_variables"] and store them as Start values
    # - Wait during the observation time
    # - Pick up latest Status values listed in the metadata["global_status_variables"] and store them as End values (to be compared with Start)
    #
    # Raises :
    # DBStatusError (on flush), StatusVarGetError
    def collectGlobalStatusValues(self, observation_time=1., complete: bool = False):
        # Almost straight on collect of Global variables
        self._getInformationSchema()
        self._dbGetStatusAtStep(step="start", statusVars=self._metadata["global_status_variables"], scope="GLOBAL")
        time.sleep(observation_time)
        self._getInformationSchema()
        self._dbGetStatusAtStep(step="end", statusVars=self._metadata["global_status_variables"], scope="GLOBAL")
        self._dbStatus["__status_collect_obs_duration"] = observation_time

        # Manage differently the collect of some Usage Status Global variables, averaged on several seconds
        # There is a problem (LOCK??) just hereafter!!!!! => DISABLED
        #self.startLatencyFromPerformanceSchema()
        self._collectUsageStatusValues(complete=complete)
        #self._dbStatus["perfschema_latency"] = self.endLatentyFromPerformanceSchema()
        
        self._dbStatus["__globvars_collected"] = True


    def getDBStatus(self):
        if self._dbStatus["__globvars_collected"] is False:
            return {}
        
        out = self._dbStatus.copy()
        #out |= self._dbStatus_init
        out["information_schema"] = {}

        for schema in self._metadata["information_schemas"]:
            out["information_schema"][schema] = self._information_schemas.get(schema)
        
        return out

    def _mapFullUsageStatus(self) -> dict:
        status = {
            "qps": self._dbStatus["qps"],
            "threads_connected": int(self._dbStatus["threads_connected_end"]),
            #"perfschema_latency": self._dbStatus["perfschema_latency"],
            "created_tmp_disk_tables_per_sec": self._dbStatus["created_tmp_disk_tables_per_sec"],
        }

        if self._dbStatus.get("db_size") is not None:
            status["db_size"] = self._dbStatus["db_size"]

        return status

    # Return usage status in a dictionary { "qps": float, "Bps": float, "KBps": str, "threads_cache_hit": int, "threads_running": int } 
    # qps stands for queries/s (actually 'Questions'  per second)
    # Bps stands for bytes_sent/s (actually 'Bytes_sent'  per second)
    # If complete is True, returns also aditional keys/dict_value as follow:
    #  "db_size": { "schema": "", "engine": "", "TableSizeMB": "xx", "IndexSizeMB": "yy"}
    #  "table_count": { "??": "xx" } 
    #  "row_count": { "??": "xx" } 
    #
    # Raises:
    # StatusVarGetError
    def getDBUsageStatus(self) -> dict:
        if self._dbStatus["__globvars_collected"] is False:
            return {}
        return self._mapFullUsageStatus()

    # Raises
    # GlobalVarError
    def _dbGetGlobalVariable(self, var: str) -> str:
        try:
            return self._dbCon.getServerVariable(var)
        except mysql.connector.Error as err:
            raise dberrors.GlobalVarError(var, err.msg)

    # Return 0 if the buffer's resize status is completed
    # For MySQL returns the value of innodb_buffer_pool_resize_status_code status variable
    def _innodb_buffer_pool_resize_status_code(self): # -> (int, str):
        if self._servername == "mariadb":
            resize_status = self._dbCon.showStatus(varname="innodb_buffer_pool_resize_status")
            if resize_status.startswith("Completed "):
                return 0, resize_status
            return 1, resize_status
        
        # 'innodb_buffer_pool_resize_status_code' and 'innodb_buffer_pool_resize_status_progress' are only available since MySQL 8.0(.31?) and not in MariaDB too
        resize_status_code = int(self._dbCon.showStatus(varname="innodb_buffer_pool_resize_status_code"))
        return resize_status_code, "..."

    # Raises:
    # KnobSetError
    # KnobUpdateInProgress when the knob cannot be updated because it an update is ongoing
    def _dbSetKnobsToDrive(self) -> None:
        for knob in self._dynKnobs.keys():
            curVal = self._dynKnobsPrevious[knob]
            newVal = self._dynKnobs[knob]
            try:
                if curVal != newVal:
                    self._dbCon.setServerVariable(knob, newVal)
                    if knob == "innodb_buffer_pool_size":
                        time.sleep(0.2)
                        resize_status_code, msg = self._innodb_buffer_pool_resize_status_code()
                        while resize_status_code != 0:
                            print(datetime.now(), "Buf pool resize in progress...", "Code:", resize_status_code, msg)
                            resize_status_code, msg = self._innodb_buffer_pool_resize_status_code()
                            time.sleep(1)
                    self._dynKnobsUpdatedInDB[knob] = True
            except mysql.connector.Error as err:
                print("EXCEPTION:", err)
                if err.errno == errorcode.ER_BUFPOOL_RESIZE_INPROGRESS:
                    time.sleep(2)
                    raise dberrors.KnobUpdateInProgress(knob, curVal, newVal)
                else:
                    raise dberrors.KnobSetError(knob, curVal, newVal)

    # Raises:
    # KnobGetError
    def _dbGetKnobsToDrive(self) -> None:
        for knob in self._dynKnobs.keys():
            try:
                self._dynKnobs[knob] = self._dbGetGlobalVariable(knob)
            except dberrors.GlobalVarError as err:    
                raise dberrors.KnobGetError(knob)
       

    def isnumber(self, str):
        try:
            float(str)
            return True
        except ValueError:
            return False
    
    # Get DB Status variables at a specific state ("init", "start" or "end")
    # If the status value is a number, compute the count (end - start values)
    # scope can be "GLOBAL", "SESSION" or ""
    # Raises:
    #  StatusVarGetError
    def _dbGetStatusAtStep(self, step="start", statusVars=[], scope: str="") -> None:
        if step == "initt" or step == "start":
            self._flushStatus()
        for sv in statusVars:
            try:
                mapping = self._information_schemas_mapping.get(sv)
                if mapping and scope.lower() == "global":
                    value = str(self._information_schemas[mapping["schema"]][mapping["name"]])
                    #print("GOT MAPPING:", sv, "IN", "information_schema."+mapping["schema"]+"."+mapping["name"], "VALUE:", value)
                else:
                    value = self._dbCon.showStatus(varname=sv, scope=scope)
                if step == "init":
                    self._dbStatus[sv+"_init"] = value
                else:
                    self._dbStatus[sv+"_"+step] = value
                    if step == "start":
                        self._dbStatus[sv+"_end"] = ""
                        self._dbStatus[sv+"_diff"] = ""
                    elif step == "end":
                        if self.isnumber(value):
                            diff = int(value) - int(self._dbStatus[sv+"_start"])
                            self._dbStatus[sv+"_diff"] = str(diff)
                        else:
                            self._dbStatus[sv+"_diff"] = self._dbStatus[sv+"_start"]+"->"+str(value)

            except mysql.connector.Error as err:
                raise dberrors.StatusVarGetError(sv, err.msg)


    # Reset statistics and global counters....
    # Raises:
    # 
    def _flushStatus(self):
        try:
            self._dbCon.flushStatement("STATUS")
        except mysql.connector.Error as err:
            raise dberrors.DBStatusError(err.msg)



    #def isGlobalStatusCollected(self) -> bool:
    #    return self._dbStatus["__globvars_collected"]    

    # Initialise context before updates
    def _beginKnobsDriving(self):
        self._dynKnobsPrevious = {}
        self._dynKnobsUpdatedInDB = {}
        for key,val in self._dynKnobs.items():
            self._dynKnobsPrevious[key] = str(int(val)) # Ensure to use string knobs only
            self._dynKnobsUpdatedInDB[key] = False

    #  
    def _endKnobsDriving(self):
        self._dynKnobsUpdatedInDB = {}
        self._dynKnobsPrevious = {}
        self._dbStatus = { "__globvars_collected": False,
                           "__complete_globavars_collected": False
                         } # Reset status after a drive (and before the collect of global values)


    #  Restore updated knobs to their previous value
    # Raise KnobRollbackError, KnobUpdateInProgress
    def _rollbackKnobsDriving(self):
        for key in self._dynKnobs.keys():
            prevVal = self._dynKnobsPrevious[key]
            if self._dynKnobsUpdatedInDB[key] == True:
                try:
                    self._dbCon.setServerVariable(key, prevVal)
                except:
                    raise dberrors.KnobRollbackError(key, self._dynKnobs[key], prevVal)
            self._dynKnobs[key] = prevVal
        self._endKnobsDriving()


    # Drive (dynamic only) database knobs to their new values.
    # Returns
    # None
    # Raises: 
    # KnobGetError
    # KnobUpdateInProgress, KnobSetError
    # KnobRollbackError 
    def driveDynamicKnobs(self, knobs={}):
        self._beginKnobsDriving()
        try:
            for k, v in knobs.items():
                self._dynKnobs[k] = str(int(v)) # Ensure to use string knobs only
            self._dbSetKnobsToDrive()
        except dberrors.KnobError as err:
            self._rollbackKnobsDriving()
            raise err

        self._endKnobsDriving()    

    def waitForDBUsage(self, threads=0) -> dict:
        low_usage = True
        while low_usage:
            try:
                tc = self._dbCon.showStatus(varname="threads_connected", scope="GLOBAL")
                #print("Threads count:", tc)
                if int(tc) >= threads:
                    low_usage = False
                else:
                    time.sleep(1.)
            except (dberrors.DBStatusError,dberrors.StatusVarGetError, mysql.connector.Error) as err:
                print(err)
                time.sleep(3.)

        print("Collect status...")
        self.collectGlobalStatusValues(observation_time=1, complete=True)
        print("Collect DONE!")
        return  self.getDBUsageStatus()         

    # Check passed values (typically the current one) with those we drove to a new direction
    # In case of discrepancy between the current values and what was expected after the last drive (may be the database did not operate well the change?),
    # return the first knob name that not match the expected value.
    # Return "" if no discrepancy is observed
    def validateDynamicKnobsLastDirection(self, currentknobs={}) -> str:
        for k in self._dynKnobs.keys():
            if currentknobs[k] != self._dynKnobs[k]:
                return k
        
        return ""


    ## An attempt to get Latency from performance schema
    ## NOT USED
    ##
    def _clean_performance_schema(self, table="events_statements_histogram_global"):
        self._dbCon.statement('truncate performance_schema.'+table+';')

    # See: https://lefred.be/content/mysql-8-0-statements-latency-histograms/
    def _execute_get_performance_schema(self, table="events_statements_histogram_global", schema_name="") -> DataFrame:
        if table == "events_statements_histogram_global":
            return self._dbCon.panda_statement('select * from performance_schema.'+table+';')
        elif table == "events_statements_histogram_by_digest":
            return self._dbCon.panda_statement("SELECT * from performance_schema.events_statements_histogram_by_digest t1 \
            JOIN performance_schema.events_statements_summary_by_digest t2 ON t2.DIGEST=t1.DIGEST AND t2.SCHEMA_NAME=t1.SCHEMA_NAME \
            where t1.SCHEMA_NAME='"+schema_name+"' AND t1.COUNT_BUCKET > 0 AND QUERY_SAMPLE_TEXT LIKE '%BEGIN%' \
            ORDER BY t1.BUCKET_NUMBER;")
        else:
            return self._dbCon.panda_statement("select * from performance_schema."+table+"where t1.SCHEMA_NAME='"+schema_name+"';")

    SQL_TIMER_FRACTION = 1000000000  # Fraction for picoseconds to milliseconds conversion

    # calculate all the latenncy (ms)
    def _calculate_latency(self, df: DataFrame) -> float:
        lat_total = 0
        lat_total_count = df["COUNT_BUCKET"].sum()

        if lat_total_count == 0:
            print("#BUCKETS<0 (PROBLEM)")
            return -1.
        
        tcount = 0
        for index, row in df.iterrows():
            count = row["COUNT_BUCKET"]
            lat_total += ((row["BUCKET_TIMER_LOW"] + row["BUCKET_TIMER_HIGH"])/2 *  count)
            tcount += count
            #print(row)
            #print("Q", row["QUERY_SAMPLE_TEXT"], "BUCKET_AVG", (row["BUCKET_TIMER_LOW"]+row["BUCKET_TIMER_HIGH"])/2, "QUANTILE", round(row["BUCKET_QUANTILE"]*100,2), "#",  count, "#QUANTILE", row["COUNT_BUCKET_AND_LOWER"])
            #print("QUERY_SAMPLE", row["QUERY_SAMPLE_TEXT"])

        lat_moy = lat_total/lat_total_count/self.SQL_TIMER_FRACTION

        print("#BUCKETS>0:", index, "LAT MOY:", lat_moy, "#Requests:", lat_total_count, tcount)
        
        return lat_moy

    # calculate latency (ms) of a period, need two dataframes
    def _calculate_latency_queries(self, df1: DataFrame, df2: DataFrame) -> float:
        lat_total1 = 0
        for index, row in df1.iterrows():
            lat_total1 += (row["BUCKET_TIMER_LOW"] + row["BUCKET_TIMER_HIGH"])/2 *  row["COUNT_BUCKET"]
        lat_total2 = 0
        for index, row in df2.iterrows():
            lat_total2 += (row["BUCKET_TIMER_LOW"] + row["BUCKET_TIMER_HIGH"])/2 *  row["COUNT_BUCKET"]
        print("DF1 SUM OF COUNT BUCKET:", df1["COUNT_BUCKET"].sum(), "LAT MOY:", lat_total1/df1["COUNT_BUCKET"].sum()/self.SQL_TIMER_FRACTION)
        print("DF2 SUM OF COUNT BUCKET:", df2["COUNT_BUCKET"].sum(), "LAT MOY:", lat_total2/df2["COUNT_BUCKET"].sum()/self.SQL_TIMER_FRACTION)
        lat_moy = ((lat_total2 - lat_total1)/(df2["COUNT_BUCKET"].sum() - df1["COUNT_BUCKET"].sum())/self.SQL_TIMER_FRACTION)
        print("LAT DF1->DF2", lat_moy)
        return lat_moy

    def startLatencyFromPerformanceSchema(self):
        self._clean_performance_schema()
        self._clean_performance_schema(table="events_statements_histogram_by_digest")
        #self._ps_histo_global1 = self._execute_get_performance_schema()

    def endLatentyFromPerformanceSchema(self) -> float:
        ps_histo_global2 = self._execute_get_performance_schema(table="events_statements_histogram_by_digest", schema_name="adbms")
        #print("******* 1 ********", self._calculate_latency(self._ps_histo_global1))
        #print(self._ps_histo_global1)
        #print("******** 2 *******", self._calculate_latency(ps_histo_global2))
        #print(ps_histo_global2)
        #print("*****************")
        #return self._calculate_latency_queries(self._ps_histo_global1, ps_histo_global2)
        return self._calculate_latency(ps_histo_global2)
