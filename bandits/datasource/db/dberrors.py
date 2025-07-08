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


class DBError(Exception):
    """Base class for other exceptions"""
    pass

class DBServerVersionError(DBError):
    """Raised when a DB server type or version is incorrect"""
    def __init__(self, curVersion="none", expectedVersion="none", message="Server type or version is not correct"):
        self.msg = message
        self.curVer = curVersion
        self.expVer = expectedVersion

    def __str__(self):
        return f'{self.msg}: Server version {self.curVer}  does not starts with {self.expVer}'

class DBNotReadyError(DBError):
    """Raised when a DB server is not ready to accept connections"""
    def __init__(self, database, message="Database is not ready to accept connections"):
        self.msg = message
        self.database = database
        super().__init__(self.msg)

    def __str__(self):
        return f'self.msg: Database {self.database} is not ready to accept connections'

class DBStatusError(DBError):
    def __init__(self, message=""):
        self.msg = message
        super().__init__(self.msg)        

class GlobalVarError(DBError):
    """Raised if a Global Variable read or write fails"""
    def __init__(self, globvar, message="Unable to read variable from database"):
        self.globvar = globvar
        self.msg = message
        super().__init__(self.msg)
    
    def __str__(self):
        return f'{self.globvar} -> {self.msg}'

class StatusVarGetError(GlobalVarError):
    """Raised when a knob can not be read correctly"""
    def __init__(self, globvar, message="Get Status Global Variable error"):
        super().__init__(globvar, message)

   
class KnobError(DBError):
    """Raised if a knob read or write fails"""
    def __init__(self, knob, message="Unable to read knob from database"):
        self.knob = knob
        self.msg = message
        #super().__init__(self.msg)
    
    def __str__(self):
        return f'{self.knob} -> {self.msg}'

class KnobGetError(KnobError):
    """Raised when a knob can not be read correctly"""
    def __init__(self, knob, message="Get knob error"):
        super().__init__(knob, message)

class KnobMetadataValueNotFound(KnobGetError):
    """Raised when matadata knob value cannot be read from database"""
    def __init__(self, knob, message="Metadata knob value not found"):
        super().__init__(knob, message)

class KnobSetError(KnobError):
    """Raised when a knob is not set correctly"""
    def __init__(self, knob, curVal, newVal, message="Set knob error"):
        super().__init__(knob, message)
        self.curVal = curVal
        self.newVal = newVal

    def __str__(self):
        return f'{self.knob} -> {self.msg} CurVal={self.curVal} NewVal={self.newVal}'

class KnobDriveError(KnobSetError):
    """Raised when a knob is driven to a wrong value (out of bounds, ...)"""
    def __init__(self, knob, curVal, newVal, message="Knob is driven to a wrong value (out of bounds, ...)"):
        super().__init__(knob,  curVal, newVal, message)

class KnobDriveWarmupError(KnobSetError):
    """Raised when a knob is driven to a wrong value (out of bounds, ...)"""
    def __init__(self, knob, curVal, newVal, message="Knob is driven while DB WARMUP is not ended since last knob change"):
        super().__init__(knob,  curVal, newVal, message)

class KnobRollbackError(KnobSetError):
    """Raised when a knob is driven to a wrong value and we were not able to rollback it to its previous value """
    def __init__(self, knob, curVal, newVal, message="Knob is driven to a wrong value and ROLLBACK failed"):
        super().__init__(knob,  curVal, newVal, message)

class KnobUpdateInProgress(KnobSetError):
    """Raised when innodb_buffer_pool_size cannot be set because last change is still in progress"""
    def __init__(self, knob, curVal, newVal, message="Knob cannot be set because last change is still in progress"):
        super().__init__(knob,  curVal, newVal, message)
