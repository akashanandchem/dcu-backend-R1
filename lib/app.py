# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:35:51 2026

@author: aanand
"""

import sys
import os


# sys.path.append(r"C:\Users\aanand\Downloads\HGI\HGI-One yr\lib")

current_dir = os.path.dirname(os.path.dirname(__file__))
#sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "lib"))




from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from hgi import module_hgi
import pandas as pd
import numpy as np
import joblib
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "DCU ML API is running"}
            
            
@app.get("/run-model")
def run_model():
    try:
        # module_hgi()  # keep commented for now
        
        # df = pd.read_csv(r"C:\Users\aanand\Downloads\HGI\HGI-One yr\output\output.csv")
        # df = df.fillna("")

        # outputFileName = os.path.join(current_dir, "output", "output.csv")
        # df = pd.read_csv(outputFileName)
        df = pd.read_csv(r"output\output.csv")
        # df = pd.read_csv(r"output.csv")
        df = df.fillna("")

        # 👉 Convert time column
        df["Time"] = pd.to_datetime(df["Time"])

        # 👉 Take last 20 rows (for trend)
        trend_df = df.tail(20)

        # 👉 Latest row
        latest = trend_df.iloc[-1]

        return {
            "latest": {
                "timestamp": str(latest["Time"]),
                "prediction": float(latest["BIL.39.CokeDrum_HGI_Pred.IDMS"]),
                "upper": float(latest["BIL.39.CokeDrum_PredHGI_Upper.IDMS"]),
                "lower": float(latest["BIL.39.CokeDrum_PredHGI_Lower.IDMS"]),
                "houronline": float(latest["HGI_DrumRunTime.HOL"]),
                "furnacecharge": float(latest["39FI123"]),
                "inlettemp": float(latest["39TI203"]),
                "inletpress": float(latest["39PI995A"]),
                "outlettemp": float(latest["39TI199"]),
                "outletpress": float(latest["39PI101"]),
                "residapi": float(latest["39DI348"]),
                "freshcharge": float(latest["39FI347"]),
                "avgcot": 45,
                "cip": 45,
                "cop": 45,
                "daysonline": 45,
                "forcastedrunlenght": 45,
                "optimizedrunlength": 45,
                "thermaleficiency": 45,
                "firedduty": 45,
                "archo2": 45,
                "stacktemperature": 45,
                "draftpressure": 45,
                "shcratio": 45,
                "damperopening": 45,
                "specificfuelconsumption": 45,
                "specificenergyconsumption": 45,
                "capacityutilization": 45,
                "cokeyield": 45,
                
                
                # ✅ ADD THIS
                "drum_status": {
                    "drum1": latest["BIL.39.HGI_D3908_Status.IDMS"],
                    "drum2": latest["BIL.39.HGI_D3909_Status.IDMS"],
                },
                
                # ✅ ADD THIS
                "furnace_status": {
                    "furnace1": 'Online',
                    "furnace2": 'Offline',
                },
                
                # ✅ ADD THIS
                "flags": {
                    "flag1": int(latest["HGIflag1"]),
                    "flag2": int(latest["HGIflag2"]),
                    "flag3": int(latest["HGIflag3"])
                },
                # ✅ ADD THIS
                "yields": {
                    "heavy": float(latest["BIL.39.CokeDrum_HeavyWtPercent.IDMS"]),
                    "medium": float(latest["BIL.39.CokeDrum_MediumWtPercent.IDMS"]),
                    "light": float(latest["BIL.39.CokeDrum_LightWtPercent.IDMS"])
                }
            },
            "trend": [
                {
                    "time": str(row["Time"]),
                    "hgi": float(row["BIL.39.CokeDrum_HGI_Pred.IDMS"]),
                    "upper": float(row["BIL.39.CokeDrum_PredHGI_Upper.IDMS"]),
                    "lower": float(row["BIL.39.CokeDrum_PredHGI_Lower.IDMS"]),
                    "houronline": float(row["HGI_DrumRunTime.HOL"]),
                }
                for _, row in trend_df.iterrows()
            ]
        }

    except Exception as e:
        return {"error": str(e)}

