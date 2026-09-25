# Listening to Quantum Tunneling

An 8-qubit model of a proton moving in a double well, with hydrogen, deuterium, and tritium on the same potential. The exact clock is a matrix exponential. A split-operator circuit runs the same grid. Two listening modes turn that run into a 16-bit WAV.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app listens on port **8741**.

Physics checks:

```bash
python tests/test_physics.py
```

## What the page shows

Experience opens first and fills the window. It plays the exact doublet for H, D, and T at the default well, with one hydrogen period stretched to 1.6 s and the other isotopes slowed by their real period ratios. |ψ|² sits on the ground-state energy, under the barrier. Tap the landscape to collapse into a well. When `src/atlas_bank.json` is present, that draw uses the Atlas acceptor population at the nearest time; otherwise it uses `src/bank.json` and the caption says the Atlas file is missing. Lab is the working bench:

- Barrier E_b from 0.1 to 0.5 eV, well half-separation a from 0.25 to 0.5 Å, and an asymmetry λ from 0 to 0.02 eV/Å. Larger λ detunes the wells and suppresses coherent tunneling.
- Isotope H, D, or T. Temperature from 100 to 400 K is only a marker on the rate sketch.
- ΔE, the tunneling period, the H/D splitting ratio, the split-operator fidelity against the exact state, and the depth of one Trotter step.
- V(x) with |ψ|² at a chosen time, donor and acceptor populations for the circuit and the exact propagator, the measurement bank's acceptor counts against the exact curve, and a log-scale sketch of k_classical(T) against a flat k_tunnel. That sketch is a qualitative illustration, not an enzyme rate. The bank plotted in Lab is the local simulator at the default hydrogen well.
- Tunneling beat: two tones whose separation follows log10(ΔE), panned by the donor and acceptor populations. This is an artistic mapping.
- Measured state: QPAM encode and decode of |ψ|² frames. The shot count is the measurement budget. Fewer shots sound noisier.
- Atlas tomography, when `src/atlas_bank.json` exists, plots the Aer acceptor population and Bloch length against the exact curve and lists the job ids. `provider` `aer` / `backend` `automatic` is a simulator. The tomography schema does not name an IBM QPU, so the page does not call this run hardware.
- Set `MOTH_API_KEY` in the environment to call the Moth API (`https://api.mothquantum.com/api/v1`, header `Authorization: Bearer`). Do not put the key in source. `src/moth_client.py` reads it at runtime. Regenerating the bank is `python src/atlas_tomography.py`. Teleblur frames are `python src/atlas_teleblur.py`.

PDB 1HED, when the download succeeds, is illustrative active-site context. The zinc–water distance is not an input to the Hamiltonian. If the download fails, the page says PDB unavailable.

## Layout

| Path | Role |
| --- | --- |
| `app.py` | Streamlit dashboard |
| `src/hamiltonian.py` | Grid, masses, sparse H = T + V |
| `src/quantum_sim.py` | Exact exponential and the split-operator circuit |
| `src/rates.py` | Qualitative Arrhenius rate, k_tunnel, and H/D ratio |
| `src/moth_sonification.py` | Tunneling beat and QPAM measured-state audio |
| `src/export_frames.py` | Precomputed H/D/T movie for the Experience view |
| `src/experience.html` | Canvas and Web Audio player for that movie |
| `src/measurement_bank.py` | 24 one-qubit prepares; local sampler and Atlas stub |
| `src/bank.json` | Shot record from the local simulator |
| `src/moth_client.py` | Moth API client. Key from `MOTH_API_KEY` |
| `src/atlas_tomography.py` | 24 `rx` circuits sent to `tomography-api-v2` |
| `src/atlas_bank.json` | Aer simulator Bloch samples, when the API run has been saved |
| `src/atlas_teleblur.py` | Uploads donor and acceptor art and calls `telablur-v1` |
| `atlas_inputs/` | `donor.png` and `acceptor.png` |
| `atlas_outputs/` | Teleblur frames and one sidecar JSON per frame |
| `src/atlas_client.py` | Older stub. The live client is `moth_client.py` |
| `src/pdb_parser.py` | Optional RCSB fetch for 1HED |
| `tests/test_physics.py` | Norm, population transfer, isotope gap, circuit fidelity |
| `tests/test_measurement_bank.py` | Local sampler label and the saved shot record |
