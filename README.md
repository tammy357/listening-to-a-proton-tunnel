# Quantum Tunneling & Sonification Engine

A Moth Hacks 2026 bench for proton transfer in human carbonic anhydrase II (PDB [1HED](https://www.rcsb.org/structure/1HED)). An 8-qubit statevector evolves in an asymmetric double well between the catalytic zinc and its bound water, and the final amplitudes are played as sound through Moth Quantum's [quantumaudio](https://quantumaudio.readthedocs.io/) QPAM encoder.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app listens on port **8741**.

## What you can do

- Read the Zn–HOH distance from RCSB. If the download fails, the page uses the offline coordinates Zn `(12.5, −8.7, 11.7)` Å and HOH `(13.9, −7.4, 15.9)` Å.
- Drag the barrier (0.1–1.0 eV), the asymmetry (0–0.2 eV), and the duration (5–50 fs).
- Watch the double well against the final proton density, and the donor versus acceptor populations.
- Press **Synthesize Quantum Audio (QPAM)** to encode the final statevector, measure it, and play a 16-bit WAV. The pitch slider is the oscillator frequency; the spatial wavefunction is one cycle. Loudness follows how much probability has reached the acceptor well.

## Physics, briefly

Internal arithmetic is SI: metres, kilograms, seconds, joules. Sliders are converted at the boundary.

The grid is 256 sites on \(x \in [-1.5, 1.5]\) Å, one site per computational-basis state of an 8-qubit register. The potential is

\[
V(x) = E_b \left(1 - (x/a)^2\right)^2 + \lambda (x/a)
\]

with well minima at \(a = 0.32\) Å. That width is short of a full hydrogen bond on purpose: with a real proton mass, a thicker barrier does not move visible probability across \(x = 0\) inside 50 fs. Positive \(\lambda\) deepens the donor side (\(x < 0\)).

The kinetic energy is the three-point stencil for \(-\hbar^2/2m\, d^2/dx^2\). The Hamiltonian is stored sparse. Time evolution applies

\[
U(\Delta t) = \exp(-i H \Delta t / \hbar)
\]

as a Qiskit `Operator` to a `Statevector`, starting from a Gaussian packet in the donor well.

At the default 0.20 eV barrier the packet's energy sits below the barrier, so acceptor population is tunneling. At the bottom of the slider (0.10 eV) the packet energy is close to the top of the barrier.

## Audio and Atlas

QPAM takes real samples in \([-1, 1]\). The complex statevector is normalised, rotated into the quadrature with the most energy, and concatenated with the orthogonal quadrature. That frame goes through `quantumaudio.encode(..., scheme="qpam")` and `quantumaudio.decode`. Reconstruction fidelity, basis-gate depth, and the Aer simulator name are written into the Atlas job card.

The card POSTs that document to the Atlas API. It does not invent a successful response.

```bash
export ATLAS_API_URL="https://api.mothquantum.com/v1/jobs"   # optional override
export ATLAS_API_KEY="..."                                    # optional bearer token
```

Without a key, the `atlas` object in the card shows the HTTP error from the public host, and the fidelity / depth / backend fields remain the local measurement.

## Layout

| Path | Role |
| --- | --- |
| `app.py` | Streamlit dashboard |
| `src/pdb_parser.py` | RCSB 1HED zinc and catalytic water |
| `src/hamiltonian.py` | Constants, grid, sparse \(H = T + V\) |
| `src/quantum_sim.py` | Gaussian packet and Qiskit time evolution |
| `src/moth_sonification.py` | QPAM encode/decode, WAV, Atlas POST |
| `.cursorrules` | SI units, 8-qubit grid, QPAM rules |
