# Energy-adaptive operation: scoping note

**For: Daniel, Hank, Kevin. Decision needed now (Week 8).** CLAUDE.md flags this
as the one proposal feature with no owner, and says to scope it by Week 8 if time is
tight. This note lays out what's feasible on the KR260 and recommends a scope.

## What the proposal asks for

"Read the Xilinx SysMon on the KR260 and adjust pipeline depth and clock speed in
response to power/thermal conditions."

## What the hardware can actually do

**Sensing**
- **SysMon measures temperature and supply voltages, not power or current.** On Linux
  it's the `xilinx-ams` IIO driver:
  - PL temperature: `in_temp20_input` (2023.1+ naming; `in_temp2_pl_temp_*` before).
  - PS temperatures and rail voltages are also available.
  - Rising-threshold events are supported.
- **Power comes from a separate sensor.** The K26 SOM carries an INA260
  current/power monitor, read through Linux hwmon. `xmutil platformstats -p` reports
  whole-SOM power, current and voltage from it.
  - This is documented for the KV260, which uses the same K26 SOM, so the KR260
    should match. **Unverified on our board:** run `xmutil platformstats -p` once to
    confirm.
  - It measures the **whole SOM** (PS + PL + DDR), not our accelerator alone.

**Actuation (what software can change at runtime)**
- **PL clock frequency:** yes.
  - The standard route is a **Clocking Wizard** IP with AXI dynamic reconfiguration,
    which has a Linux driver.
  - The PS→PL clock (`pl_clk0`) can also be retargeted through the Linux clock
    framework, e.g. the out-of-tree `fclkcfg` driver.
  - Either way, the accelerator must be **idle** while the clock changes.
- **Pipeline depth: no.** In HLS it's fixed at synthesis. Changing it at runtime
  would mean swapping bitstreams with partial reconfiguration (DFX): weeks of work,
  not days.
- **Already in our design, zero hardware cost:**
  - **Sparsity operating point:** weights are reloaded on every `dsc_top` call, so
    switching between Kevin's 30/50/70%-pruned weight sets is just pointing the driver
    at a different parameter buffer.
  - **`skip_en`:** skip on/off.
  - **Frame pacing:** how often the driver starts an inference.
- **Core voltage: out of scope.** The K26 is a -2LV part at fixed VCCINT; changing
  regulator voltages via PMBus at runtime is risky and undocumented for this board.

**Timing headroom:** `dsc_top` is estimated at ~274 MHz in HLS against our 200 MHz
target. So 100/150/200 MHz should be safe, and 250 MHz is worth trying once Vivado
implementation gives real post-route timing.

## The physics, so we claim the right thing

- **Slowing the clock does not reduce energy per inference.** Without voltage
  scaling, dynamic energy per operation is roughly fixed, and a slower clock just
  stretches the run, so static energy per inference goes *up*. Energy per inference
  is usually lowest at the highest stable clock ("race to idle").
- **What adaptation genuinely buys is power and thermal capping:** holding the SOM
  under a power or temperature limit by trading throughput (clock, frame rate) or
  accuracy (sparsity level).
- **Sparsity is the one knob that lowers energy per inference:** fewer MAC cycles
  means less time and less switching for the same work. That ties the feature to
  our headline result.

**Measurement caveat:** our accelerator is a small slice of the SOM, ~10% of LUTs.
Its power delta may be small next to the SOM total, and the INA260's power reading
moves in ~10 mW steps. Measure energy over many back-to-back inferences (seconds,
not one 0.36 ms call), and subtract an idle baseline.

## Options

| | Scope | Hardware change | Effort | Risk |
|---|---|---|---|---|
| **A. Measure only** | Report power, temperature and energy per inference at each sparsity point and clock. Already Kevin's Week 10 plan. | None | Small (already planned) | Low |
| **B. Software governor** *(recommended)* | ARM-side control loop: read SysMon PL temperature + INA260 power every N ms, and step between operating points (sparsity level, frame pacing) to stay under a temperature/power cap. | None | ~2–3 days, driver side | Low |
| **C. B + clock scaling** | Also switch the PL clock (e.g. 100/150/200 MHz) via a Clocking Wizard, idling the accelerator around each change. | Clocking Wizard + reset/idle handshake in the block design; timing closure at the top frequency | +3–5 days across block design and driver | Medium (clock changes under load, timing closure) |
| **D. Runtime pipeline depth** | As literally proposed. | Partial reconfiguration (DFX), multiple kernel variants | Weeks | High |

## Recommendation

**Do B. Add C only if B is working by early Week 10. Report D as future work,** and
state plainly in the report that pipeline depth is fixed at synthesis in HLS, so
runtime adaptation uses clock, sparsity and pacing instead.

Why B:
- It needs **no hardware change**, so it can't put the bitstream at risk in Week 9.
- It reuses what we've already built: the sparsity operating points, `skip_en`, and
  `sched_len` as a readback check that the weight set really switched.
- It produces a clear demo: temperature and power over time with the governor on vs
  off, plus the throughput/accuracy cost of holding a cap.

**Suggested ownership** (sensing and power profiling are already in Kevin's scope):

| Who | Option B | Option C, if pursued |
|---|---|---|
| **Kevin** | Governor loop and SysMon/INA260 reading in the ARM driver; experiments | Clock-switch calls in the driver |
| **Hank** | Nothing | Clocking Wizard in the block design, address map, safe-switch handshake |
| **Daniel** | Nothing new in hardware; help pick operating points from the skip-rate data, and explain the results | Report post-route Fmax for the top frequency |

## Decisions needed from the team

1. Scope: **B** (recommended), A only, or B+C?
2. Owner for the governor: Kevin, as suggested?
3. The cap to demonstrate: a temperature threshold, a power threshold, or both?
4. Someone with the board: run `xmutil platformstats -p` and check the AMS sysfs
   nodes, to confirm the sensors on our KR260 and Ubuntu image.

## Sources

- AMD wiki, *Zynq UltraScale+ MPSoC AMS* (SysMon driver, sysfs names, 2023.1 rename):
  https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18842163/Zynq+UltraScale+MPSoC+AMS
- AMD wiki, *Programmable Logic Power Management* (Clocking Wizard dynamic
  reconfiguration, PS→PL clocks):
  https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/1433960485/Programmable+Logic+Power+Management
- AMD wiki, *Controlling FCLKs in Linux*:
  https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18841795/Controlling+FCLKs+in+Linux
- `fclkcfg` (user-space PL clock configuration): https://github.com/ikwzm/fclkcfg
- element14, *XMUTIL… understanding platformstats* (INA260/AMS power readout):
  https://community.element14.com/members-area/personalblogs/b/blog/posts/xmutil-understanding-platformstats
- Linux `ina2xx` hwmon driver (INA260 units): https://docs.kernel.org/hwmon/ina2xx.html
