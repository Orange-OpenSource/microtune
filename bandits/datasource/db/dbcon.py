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
import mysql.connector
from mysql.connector import errorcode
import pandas as pd
from pandas import DataFrame


class DBCon():
    # See https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlconnection.html
    # Config: host, port, user, password, database, ...
    def __init__(self, config={}):

        try:
            self.cnx = mysql.connector.connect(**config)

        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)
#        else:
#            self.cnx.close()

    def isConnected(self) -> bool :
        return self.cnx.is_connected()

    def close(self):
        self.cnx.close()

    # Options, see: https://dev.mysql.com/doc/refman/8.0/en/flush.html
    # ENGINE LOGS
    # ERROR LOGS
    # LOGS
    # STATUS (default)
    def flushStatement(self, option:str="STATUS"):
        query="FLUSH NO_WRITE_TO_BINLOG "+option+";"
        cursor = self.cnx.cursor(buffered=False)    
        cursor.execute(query)
        cursor.close()

    # Options, see: https://dev.mysql.com/doc/refman/5.7/en/reset.html
    # MASTER (default)
    # QUERY CACHE
    # SLAVE
    def reset(self, option:str="MASTER"):
        query="RESET "+option+";"
        cursor = self.cnx.cursor()    
        cursor.execute(query)
        cursor.close()

    # Get server version
    # Returns { version: "xxx" }
    def version(self) -> dict:
        query="SELECT VERSION() as version;"
        return self.fetchOne(query)

    # scope can be "GLOBAL", "SESSION" or ""
    def showStatus(self, varname="", scope="GLOBAL"):
        cursor = self.cnx.cursor()
        value = ""
    
        cursor.execute("SHOW "+scope+" STATUS LIKE '"+varname+"';")
        row = cursor.fetchone()
        if row is not None:
            value = row[1]
        cursor.close()
        
        return value

    def setServerVariable(self, variable:str, value:str):
        cursor = self.cnx.cursor()    
#        query = (
#            "SET GLOBAL "+variable+"=%d;"
#        )
#        cursor.execute(query, (value))
        cursor.execute("SET GLOBAL "+variable+"="+value+";")
        cursor.close()

    def getServerVariable(self, variable:str) -> str:
        cursor = self.cnx.cursor()
        value = ""
    
        cursor.execute("SELECT @@GLOBAL."+variable+";")
        row = cursor.fetchone()
        if row is not None:
            value = row[0]
        cursor.close()
        
        return value
    
    def statement(self, query:str):
        cursor = self.cnx.cursor()
        cursor.execute(query)
        cursor.close()

    def fetchAll(self, query:str, dictionary: bool=False):
        cursor = self.cnx.cursor(buffered=True, dictionary=dictionary)
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()

        return data

    def fetchOne(self, query:str) -> dict:
        cursor = self.cnx.cursor(dictionary=True)
        cursor.execute(query)
        row = cursor.fetchone()
        cursor.close()

        return row

    def panda_statement(self, query:str) -> DataFrame:
        return pd.read_sql(query, self.cnx)


