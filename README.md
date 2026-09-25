# ⚡ Quantum Proton Tunneling & Creative Sonification — Moth Hacks 2026

Simulate quantum hydrogen tunneling in the active site of **Human Carbonic Anhydrase II**
(RCSB PDB [`1HED`](https://www.rcsb.org/structure/1HED)) on an **8-qubit register**, then
**listen to the statevector** through Moth Quantum's [`quantumaudio`](https://pypi.org/project/quantumaudio/)
**QPAM** encoding.

Carbonic anhydrase II hydrates CO₂ at up to ~10⁶ turnovers per second, and the step that
limits it is not the CO₂ chemistry — it is handing a proton off from the zinc-bound water. The
proton is light enough to **tunnel** through the activation barrier instead of climbing over
it, which is why the reaction shows a large kinetic isotope effect. This app solves that
proton's Schrödinger equation and turns the quantum dynamics into sound.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (default <http://localhost:8501>). No API keys, no
database, no network required — if RCSB is unreachable the app falls back to the deposited
1HED reference coordinates and keeps running.

## What you can do in the app

| Control | Effect |
| --- | --- |
| **Barrier Height** `E_b` (0.1–1.0 eV) | Raises the barrier between donor and acceptor wells. Transfer dies off exponentially. |
| **Asymmetry Drive** `λ` (0.0–0.2 eV) | Tilts the landscape — the thermodynamic driving force. `λ = 0` gives coherent back-and-forth oscillation. |
| **Simulation Duration** (5–50 fs) | How much quantum dynamics to propagate. |
| **Base Audio Pitch** (55–880 Hz) | Fundamental of the QPAM wavetable oscillator. |
| **Proton ¹H / Deuteron ²H** (advanced) | The in-app kinetic isotope effect: double the mass and watch tunneling collapse. |
| **QPAM shots per frame** (advanced) | 20000 shots = clean tone; 500 shots = audible quantum shot noise. |
| **🎵 Synthesize Quantum Audio (QPAM)** | Encodes the trajectory through `quantumaudio` and renders a playable WAV. |
| **🌐 Moth Atlas Platform API Output** | The job document (fidelity, circuit depth, active backend) for the Atlas platform. |

## How it works

1. **`src/pdb_parser.py`** downloads `1HED.pdb` from the RCSB web API, extracts `ZN A 262`
   and the catalytic water `HOH A 342`, and reports their separation in Ångströms. The
   download is cached in `data/`; a hardcoded fallback keeps the demo alive offline.
2. **`src/hamiltonian.py`** builds a 256-point spatial grid over `x ∈ [-1.5 Å, +1.5 Å]`
   (256 = 2⁸, i.e. one grid point per computational basis state of **8 qubits**), the
   asymmetric double-well potential

   ```
   V(x) = E_b · (1 − (x/a)²)²  +  λ · (x/a)
   ```

   and the sparse Hamiltonian `H = T + V`, where `T = −(ħ²/2m_H)·d²/dx²` is a 2nd-order
   central finite-difference Laplacian. Everything internal is **SI units**.
3. **`src/quantum_sim.py`** prepares a Gaussian wavepacket in the donor well (`x < 0`),
   exponentiates the Hamiltonian into `U(dt) = exp(−iH·dt/ħ)`, wraps it in a Qiskit
   `UnitaryGate`, and applies it step by step to a `qiskit.quantum_info.Statevector`. The
   propagator is exact in `dt` and unitary to machine precision, so norm fidelity stays at
   1.0 to ~1e-12. Donor (`x < 0`) and acceptor (`x > 0`) populations are accumulated over
   `t ∈ [0, 50 fs]`.
4. **`src/moth_sonification.py`** projects each complex statevector onto a real 256-sample
   audio frame (using `Re ψ`, so the **quantum phase** survives), then

   ```python
   circuit = quantumaudio.encode(frame, scheme="qpam")   # 256 samples -> 8 qubits
   samples = quantumaudio.decode(circuit, shots=8000)    # measure it back
   ```

   QPAM (Quantum Probability Amplitude Modulation) stores sample `k` in the *probability* of
   basis state `|k⟩`, so a 256-sample frame needs exactly the same 8 qubits the physics uses.
   Decoded frames become single-cycle **wavetables**; crossfading them stretches a ~50 fs
   quantum event by ~10¹³ into the audio band, while the pitch glides up with the
   acceptor-well population. Output is 16-bit PCM mono WAV in an `io.BytesIO` buffer for
   `st.audio()`.
5. **`app.py`** is the Streamlit dashboard: active-site metrics, Plotly subplots (potential +
   wavepacket density, and population dynamics), a time-scrubbing slider, the QPAM audio
   player, and the Moth Atlas job card.

Because decoding QPAM is a *measurement*, the reconstruction is shot-noise limited
(error ~ `1/√shots`). That noise is deliberately left in the audio — it is the sound of
sampling a quantum state.

## Project layout

```
.
├── app.py                     # Streamlit dashboard (UI only)
├── requirements.txt
├── .cursorrules               # SI units, 8-qubit/256-grid contract, QPAM standards
├── .streamlit/config.toml     # dark theme
└── src/
    ├── pdb_parser.py          # RCSB PDB 1HED active-site parser
    ├── hamiltonian.py         # SI constants, grid, V(x), sparse H = T + V
    ├── quantum_sim.py         # Gaussian wavepacket + Qiskit unitary propagation
    └── moth_sonification.py   # quantumaudio QPAM encode/decode, WAV, Atlas payload
```

Every module has a `__main__` smoke test:

```bash
python -m src.pdb_parser        # active-site geometry
python -m src.hamiltonian       # grid, eigenvalues, tunneling splitting
python -m src.quantum_sim       # populations, norm fidelity
python -m src.moth_sonification # QPAM round-trip -> quantum_tunneling.wav
```

## Moth Atlas platform

The Atlas card always shows the exact job document for the run: statevector norm fidelity,
QPAM encode/decode fidelity, shot-noise floor, and the real hardware cost of the QPAM circuit
(its depth and CX count after transpilation to a `{cx, u}` basis — ~497 deep with 247 CX for a
256-sample frame, versus an abstract depth of 2 before decomposition).

With no credentials the QPAM circuits execute on the local simulator and the card says so. To
dispatch the same document to Atlas:

```bash
export MOTH_ATLAS_API_KEY="…"                 # or MOTH_API_KEY
export MOTH_ATLAS_API_URL="https://…/v1/jobs" # optional endpoint override
streamlit run app.py
```

## Things worth trying for the demo

- Set **Asymmetry Drive** to `0.00` and watch the symmetric well: the transfer stalls because
  the bare tunneling splitting is sub-meV, i.e. the coherent oscillation period is picoseconds.
- Push **Barrier Height** to `1.0 eV`: the wavepacket is trapped, the timbre stays static.
- Switch to **Deuteron (²H)** at fixed barrier: transfer collapses. That is the kinetic isotope
  effect, and it sounds like the pitch glide flattening out.
- Drop **QPAM shots** to `500`: the same quantum state, now audibly noisy.

## References

- Moth Quantum, `quantumaudio` — QPAM and related quantum audio representations.
- Itoh, Hara & Fujimoto (1996), *1HED*: Human carbonic anhydrase II, RCSB PDB.
- Silverman & McKenna (2007), *Solvent-mediated proton transfer in catalysis by carbonic anhydrase*.
- Klinman & Kohen (2013), *Hydrogen tunneling links protein dynamics to enzyme catalysis*.
