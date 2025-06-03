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
import plotly.express as px
from plotly.subplots import make_subplots
#from plotly.graph_objects import Figure

from bandits.perf_meter import PerfMeter

class GraphPx():
    def __init__(self, x=[], title="", title_x="X"):
        self.title = title
        #xi = [idx for idx in range(0,len(x))]
        # TODO: another type checking...
        #if len(x) > 10 and type(x[0]) != type(0):
        #    print(f'IS STR !! LEN{len(x)} type:{type(x[0])}')
        #    self.x = xi
        #    self.x_labels = x
        #else:
        #    self.x_labels = xi
        self.x = x
        self.title_x = title_x
        self.titles_y = {}
        self.y = []
        self.start_at = 0
        self._perfs = []
        self._show_perfs = False
        self.pie_data = None
        self.pie_labels = None
        self.threshold = None
        self.width = 2000

    def addCurve(self, name="", y=[], perf=None):
        self.titles_y["wide_variable_"+str(len(self.titles_y))] = name
        self.y.append(np.array(y[self.start_at:]))
        if perf is not None:
            self._perfs.append(perf)
            self._show_perfs = True if min(self._perfs) != 0 else False
    
    def setPieData(self, data, threshold):
        less_than_zero = sum(abs(x) for x in data if x < 0)
        between_zero_and_threshold = sum(x for x in data if 0 <= x <= threshold)
        greater_than_threshold = sum(x for x in data if x > threshold)
        self.threshold= threshold
        self.pie_data = [between_zero_and_threshold, less_than_zero, greater_than_threshold]
        self.pie_labels = [f'Total Idelta cumulated 0 <= idelta <= {threshold} (SLA OK)', 'Total Idelta cumulated idelta < 0 (SLA VIOLATION)', f' Total Idelta cumulated idelta > {threshold} (SLA OVER)']

    def figure(self):
        # Line chart
        p_title=''
        if self._show_perfs:
            p_min = min(self._perfs)
            p_title = f'PI: {[round(x/p_min,2) for x in self._perfs]}'

        y = self.y.copy() # Don't know why a copy is NECESSARY, else y elements are consumed by next px.line instruction and 2nd call to figure() will abort ^^
        fig = px.line(x=self.x[self.start_at:], labels=self.x_labels, y=y, width=self.width)
        fig.update_layout(
            title=self.title+f' {p_title}',
            xaxis_title=self.title_x,
            yaxis={"title":"", "fixedrange":False},
            legend={"title": "", "itemclick":'toggle'},
            font=dict(
                size=16,
                color="RebeccaPurple"
            ),
        )
        fig.for_each_trace(lambda t: t.update(name = self.titles_y[t.name],
                                            legendgroup = self.titles_y[t.name],
                                            hovertemplate = t.hovertemplate.replace(t.name, self.titles_y[t.name])
                                            )
                        )

        return fig        

    def getPieFigure(self):
        p_title = ''
        if self._show_perfs:
            p_min = min(self._perfs)
            p_title = f'PI: {[round(x/p_min, 2) for x in self._perfs]}'
        
        color_map = {
            f'Total Idelta cumulated 0 <= idelta <= {self.threshold} (SLA OK)': 'green',
            'Total Idelta cumulated idelta < 0 (SLA VIOLATION)': 'red',
            f' Total Idelta cumulated idelta > {self.threshold} (SLA OVER)': 'blue'
        }
        
        pie_fig = px.pie(values=self.pie_data, names=self.pie_labels, color=self.pie_labels, color_discrete_map=color_map)
        
        pie_fig.update_layout(
            title=self.title + f' {p_title}',
            font=dict(size=12, color="RebeccaPurple"),
            width=800,
            height=500 
        )
        
        return pie_fig

        
    # to make everything coehrent
    def figure(self):
        if self.pie_data is not None:
            # Create a subplot with 2 rows and 1 column, adjusting the row heights
            fig = make_subplots(rows=2, cols=1, 
                                subplot_titles=(self.title + " Line Chart", self.title + " Pie Chart"),
                                row_heights=[0.5, 0.5],  # Adjust the relative sizes here
                                specs=[[{"type": "xy"}], [{"type": "domain"}]],
                                vertical_spacing=0.2)  # Adjust vertical spacing here

            # Line chart
            p_title = ''
            if self._show_perfs:
                p_min = min(self._perfs)
                p_title = f'PI: {[round(x/p_min, 2) for x in self._perfs]}'

            y = self.y.copy()
            line_fig = px.line(x=self.x[self.start_at:], y=y)
            line_fig.for_each_trace(lambda t: t.update(name=self.titles_y[t.name],
                                                    legendgroup=self.titles_y[t.name],
                                                    hovertemplate=t.hovertemplate.replace(t.name, self.titles_y[t.name])
                                                    )
                                    )

            # Add line chart to the first subplot
            for trace in line_fig.data:
                fig.add_trace(trace, row=1, col=1)

            # Pie chart with custom colors
            if self.pie_data is not None:
                color_map = {
                    f'Total Idelta cumulated 0 <= idelta <= {self.threshold} (SLA OK)': 'green',
                    'Total Idelta cumulated idelta < 0 (SLA VIOLATION)': 'red',
                    f' Total Idelta cumulated idelta > {self.threshold} (SLA OVER)': 'blue'
                }
                pie_fig = px.pie(values=self.pie_data, names=self.pie_labels, color=self.pie_labels, color_discrete_map=color_map)
                for trace in pie_fig.data:
                    fig.add_trace(trace, row=2, col=1)

            # Update layout
            fig.update_layout(
                title=self.title + f' {p_title}',
                xaxis_title=self.title_x,
                yaxis_title="",
                font=dict(size=12, color="RebeccaPurple"),
                legend={"title": "", "itemclick": 'toggle'},
                width=self.width,
                height=700 
            )

            return fig
        else:
            p_title=''
            if self._show_perfs:
                p_min = min(self._perfs)
                p_title = f'PI: {[round(x/p_min,2) for x in self._perfs]}'

            y = self.y.copy() # Don't know why a copy is NECESSARY, else y elements are consumed by next px.line instruction and 2nd call to figure() will abort ^^
            fig = px.line(x=self.x[self.start_at:], y=y, width=self.width)
            fig.update_layout(
                title=self.title+f' {p_title}',
                xaxis_title=self.title_x,
                yaxis={"title":"", "fixedrange":False},
                legend={"title": "", "itemclick":'toggle'},
                font=dict(
                    size=16,
                    color="RebeccaPurple"
                ),
            )
            fig.for_each_trace(lambda t: t.update(name = self.titles_y[t.name],
                                                legendgroup = self.titles_y[t.name],
                                                hovertemplate = t.hovertemplate.replace(t.name, self.titles_y[t.name])
                                                )
                            )

            return fig        
    
import pandas as pd

# A perfmeter named "oracle" is mandatory
class GraphPerfMeterComparaison():
    def __init__(self, oracle_perf_meter: PerfMeter=None, perf_meter_list: PerfMeter=[], title="", stage=""):
        self.oracle_pf = oracle_perf_meter
        self.pf_list = perf_meter_list
        self.title = f'Perf vs. Oracle ref-{title}-Stage: {stage}'
        self.width = 2000

    def _figure(self, df: pd.DataFrame):
        # Create scatter plot
        fig = px.scatter(
            df,
            x="under_sla_tot",
            y="ram_quantities_tot",
            color='Name',
            title=self.title,
            labels={"scalar_perf": 'Scalar Perf', 'ram_quantities_tot': 'Cumulated RAM', 'under_sla_tot': '#Under SLA', 'violations_tot': '#SLA violations', 'label': "Info", 'type': "baseline", 'episodes': "#Episodes", 'steps': "#Steps"},
            hover_data=['scalar_perf', 'violations_tot', 'label', 'episodes', 'steps', 'stage'],
#            symbol='type',
            width=self.width
        )

        oracle_x = self.oracle_pf.under_sla_tot  # Extract scalar value
        oracle_y = self.oracle_pf.ram_quantities_tot  # Extract scalar value

        # Add vertical and horizontal lines for Oracle values
        fig.add_shape(
            type='line',
            x0=oracle_x, x1=oracle_x,
            y0=df['ram_quantities_tot'].min(), y1=df['ram_quantities_tot'].max(),
            line=dict(color='red', width=2, dash='dash')
        )
        fig.add_shape(
            type='line',
            x0=df['under_sla_tot'].min(), x1=df['under_sla_tot'].max(),
            y0=oracle_y, y1=oracle_y,
            line=dict(color='blue', width=2, dash='dash')
        )

        return fig

    def figure(self):
        data = {
            "Name": [],
            "trial": [],
            #"Reward": [77.51, 10.52, 82.99, 84.49, 16.09, 71.10, 82.13, 76.32, 95.75, 22.50, 6.66],
            #"Version": ["v2", "v3", "v4", "v3", "v3", "v1", "v2", "v1", "v2", "v4", "v4"],
            #"Test/Eval": ["Eval", "Test", "Test", "Eval", "Eval", "Test", "Eval", "Test", "Test", "Test", "Eval"],
            #"move_idelta": [-3.83, -5.08, -6.12, 2.28, -4.26, -0.65, -0.61, -0.25, 1.47, -0.76, -4.28],
            #"Seed": [303, 102, 703, 200, 18, 620, 947, 98, 113, 713, 324],
            "scalar_perf": [],
            "under_sla_tot": [],
            "ram_quantities_tot": [],
            "violations_tot": [],
            "label": [],
            "type": [],
            "stage": [],
            "episodes": [],
            "steps": [],
        }

        assert self.oracle_pf and self.oracle_pf.isOracle, f'Error, no Oracle Perfmeter is provided with Oracle:{self.oracle_pf} and List: {data["Name"]}...'

        pf_list = []
        pf_list.extend(self.pf_list)
        pf_list.append(self.oracle_pf)

        for pm in pf_list:
            data["Name"].append(pm.name)
            data["trial"].append(pm.trial)
            data["scalar_perf"].append(pm.getSessionScalarPerformance(oracle_perf_meter=self.oracle_pf))
            data["under_sla_tot"].append(pm.under_sla_tot)
            data["ram_quantities_tot"].append(pm.ram_quantities_tot)
            data["violations_tot"].append(sum(pm.violations_count_per_ep))
            data["label"].append(pm.label)
            data["type"].append(pm.baseline) #.append(int(pm.isOracle))
            data["stage"].append(pm.stage)
            data["episodes"].append(pm.getSessionEpisodesCount())
            data["steps"].append(pm.getSessionStepsCount())

        df = pd.DataFrame(data)

        return self._figure(df)

class GraphPredictionsStatus():
    def __init__(self, title="", sla_threshold=20):
        self.title = title
        self.sla_threshold = sla_threshold
        self.width = 2000

    def _figure(self, df: pd.DataFrame):
        violations = df['state'].value_counts().get("VIOLATION", 0)
        over = df['state'].value_counts().get("OVER", 0)
        ok = df['state'].value_counts().get("OK", 0)
        under = df['state'].value_counts().get("UNDER", 0) + violations
        color_discrete_map = {'OVER': 'blue', 'UNDER': 'orange', 'OK': 'green', 'VIOLATION': 'red'} # yellow, blueviolet, grey, gold, darkcyan, maroon, deeppink

        # Create scatter plot
        fig = px.scatter(
            df,
            x="step",
            y="latency_ms",
            color='state',
            color_discrete_map=color_discrete_map,
            title=f'{self.title} #UNDER: {under} with #VIOLATIONS:{violations} #OVER: {over} OK: {ok}',
            labels={'step': 'Steps of predictions on all workloads', 
                    'state': 'SLA state', 
                    'latency_ms': 'Latency (ms)', 
                    'buffer_mb': 'Buffer size (MB)', 
                    'step_in_wl': 'Num step in cur workload'},
            hover_data=['step', 'latency_ms', 'state', 'buffer_mb', 'step_in_wl'],
#            symbol='type',
            width=self.width
        )

        # Add horizontal lines for Oracle values
        fig.add_shape(
            type='line',
            x0=0, x1=df['step'].count(),
            y0=self.sla_threshold, y1=self.sla_threshold,
            line=dict(color='blue', width=2, dash='dash')
        )

        return fig

    def figure(self, states=[], latencies=[], buffers=[], step_in_wl=[]):
        assert len(states) == len(latencies)

        data = {
            "step": [idx for idx in range(0,len(states))],
            "state": states,
            "latency_ms": latencies,
            "buffer_mb": buffers,
            "step_in_wl": step_in_wl,
        }

        df = pd.DataFrame(data)

        return self._figure(df)
