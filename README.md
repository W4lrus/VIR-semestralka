# VIR-semestralka

### HOW TO: Switch main display rendering off
Open settings.json file from Documents/Airsim and add the following option:
  ```json 
  "ViewMode": "NoDisplay"
  ```
Than you can start AirSim simulation and the screen should be black.

### HOW TO: speed up simulation speed
Open settings.json file from Documents/Airsim and add the following option:
```json
 "ClockSpeed": 1
```
1 is the clock speed multiplier. You should be able to set it to any positive number. A value too large might affect simulation accuracy though.

See [settings](https://microsoft.github.io/AirSim/docs/settings/#viewmode) for more info.
