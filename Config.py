import json 
class SimulationParameters():
    def __init__(self, JsonFile) -> None:
        with open(JsonFile) as f:
            self.SimulationParameters = json.load(f).get("SimulationParameters")
    def Configure(self):
        self.TotalCPUCapacity = self.SimulationParameters.get("TotalCPUCapacity")
        self.NumberOfPS = self.SimulationParameters.get("NumberOfPS")
        self.NumberOfTCh = self.SimulationParameters.get("NumberOfTCh")
        self.NumberOfBits = self.SimulationParameters.get("NumberOfBits")
        self.NumberOfCpuCycles = self.SimulationParameters.get("NumberOfCpuCycles")
        self.windows = self.SimulationParameters.get("windows")
        self.startupdate = self.SimulationParameters.get("startupdate")
        self.deadlines = self.SimulationParameters.get("deadlines")
        self.AoI_sensitivity = self.SimulationParameters.get("AoI_sensitivity")



        