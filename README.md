# Listening to a proton tunnel

A proton in a model double well, watched and heard. Hydrogen, deuterium, and tritium share one potential. The exact clock is a matrix exponential. A split-operator circuit runs the same 8-qubit grid. Moth's engines measure that clock, blur a picture of the crossing, colour the state, and give the rhythm a room.

## Pipeline

Each engine does one job, and each job follows the physics.

**Measure.** The exact wave ψ(x, t) is reduced to a one-qubit clock, `rx(2π t / T)` on `|0⟩`. Atlas Tomography returns a Bloch vector at 24 moments. Those vectors are the tap-to-observe bank, drawn beside the ideal acceptor curve.

**See.** Donor and acceptor pictures go through Teleblur for the intro. The quantum carpet, position across and time down, goes through Tessa and becomes the cover.

**Hear.** The tunneling curve is the rhythm. Retrocausal Echo plays the three crossings. QRC MIDI continues that rhythm, Blur Jazz scatters it, and a second echo is the coda.

**Develop.** At a quarter period, position measurements of `|ψ|²` are developed as silver grains. A contact sheet holds H, D, and T at 64, 256, 1024, and 4096 grains. A ten-second film lets them accumulate.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8741 --server.address 0.0.0.0
```

The notebook reads the saved banks and does not call the API unless `MOTH_API_KEY` is set:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/moth_tunneling_workflow.ipynb
jupyter nbconvert --to html --HTMLExporter.theme=dark notebooks/moth_tunneling_workflow.ipynb
```

Physics checks: `python tests/test_physics.py`.

## Hardware, once a QPU provider is granted

`tomography-api-v2` (moth-api v0.41.0) takes `provider_name` and `backend_name` as free strings. The schema names the defaults `aer` and `automatic`. It has no enum and no IBM device list. Pass whatever provider and backend Moth has granted:

```bash
MOTH_API_KEY=your-key python src/atlas_tomography.py --backend PROVIDER/BACKEND
```

`--backend aer` is the simulator default (`aer` / `automatic`). Every run writes `src/atlas_bank_<provider>_<backend>.json` and leaves `src/atlas_bank.json` untouched. The Experience, the Lab, and notebook section 03 plot every bank that is on disk, beside the ideal curve. Until a bank is marked hardware, the row reads `hardware run: pending QPU access`.

## Engines and job ids

All of these ran on simulators. `aer` is a noiseless simulator. `fake_fez` is Emulated IBM Fez, a noise model, not a QPU.

| Engine | Job | What it did |
| --- | --- | --- |
| `tomography-api-v2` | 24 jobs, `aer` / `automatic`, 4096 shots. First `86ca6264-75fb-465c-9340-40608f455667` | One-qubit Bloch vectors. The full list is in `src/atlas_bank.json` |
| `telablur-v1` | `943710cf-9301-4476-b79f-f38df9fccf0c` | Intro frame, strength 0.5, both shapes |
| `retrocausal-echo-v1` | `484db3ab-4a72-4c55-9e69-bd26ac3246e8` | The three movements, `aer`, exact |
| `qrc-midi-v1` | `b6f554d7-089a-4204-9841-ef4ec8e8a25e` | Sixteen-bar continuation |
| `blur-midi-v1` | `3ce48680-cf94-47e7-9f38-369903d7ce21` | Scattered continuation |
| `retrocausal-echo-v1` | `c4236288-d0a0-4728-9450-c4ecf2f18710` | Coda echo |
| `tessa-image-v1` | `a4e9e559-9c6e-4a7b-ad6a-98b27f13dd3a` | Cover carpet, `aer`, 4096 shots |
| `tessa-image-v1` | `de338f15-ceda-4856-8317-56376c56cab6` | Carpet, `fake_fez` |
| `tessa-image-v1` | `985b2d3a-d29a-4a11-bac8-190a7ae3a924` | Colour strip, `aer` |
| `tessa-image-v1` | `65ae181e-9cfd-4cac-8773-6b134fc1dcea` | Colour strip, `fake_fez` |

The other Teleblur frames, from the strength sweep and the earlier morph, are in `atlas_outputs/`.

Tomography job ids, in time order:

`86ca6264-75fb-465c-9340-40608f455667`, `f7b0c643-471d-48d8-9e11-35aab6890640`, `1e1f8c75-805c-4136-9d48-a3e8076bd91d`, `332f4da2-5be6-4331-9321-f5f52af74d07`, `5eba3feb-3bf9-4abd-b016-3948f0c5faef`, `dff27734-1b5a-4102-8e2b-936d0082a68e`, `3899defe-3d2f-417d-ab78-13892240cf36`, `932a4227-2872-4ec7-bbc3-2bed2e20f170`, `b18ddbde-1dc3-4f28-90af-f655a3d30931`, `27b0892b-be77-4f3c-8e89-046bd646aa93`, `44280f4f-0160-4c8f-9e04-3111c1e50a82`, `98658466-298a-4c1c-9215-c481ead651d2`, `6ea83342-a74f-4ee6-957c-01f7bdf177bc`, `b63752a4-d4bb-4789-9fb5-8ef800a77414`, `ff34084a-a5ea-43b3-aace-f40cfe8d79a1`, `529dfa64-2a80-4c5f-a952-1ee289e43f3e`, `660a1729-94ea-40a1-8e88-4986cf8f4456`, `3cb5a9bd-0f73-4ed6-924d-3885eae43786`, `7f71b375-313b-4ba6-b883-8538a84277d0`, `54854431-fa29-409b-96c9-2bd9d951f1af`, `dbff13bd-8a08-4e98-8c82-81837791e3aa`, `6e4e3cc7-6755-4fd9-9bb2-d8f207f8da62`, `2235d52f-6410-4eaf-aed8-9f459ee7cde7`, `5819e8f4-e919-4076-9273-f6c8821ef08c`.

## Credits

Through the Wall, cover and pipeline by Lily S. Engines by Moth; Retrocausal Echo by Spencer Topel.

## What this is and isn't

This is a model double well, barrier 0.15 eV, wells 0.35 Å apart, on 256 grid sites and 8 qubits. The isotope slowing is the real splitting ratio of that well: deuterium about 5.3 times, tritium about 19.3 times. The sound and the pictures are mapped from that physics.

It is not an enzyme rate. Soybean lipoxygenase and alcohol dehydrogenase are the reason a proton tunnel is worth listening to. PDB 1HED, when it loads, is a picture of an active site. The zinc–water distance is not an input to the Hamiltonian.

It is not a hardware coherence plot. The saved tomography bank is Aer, a simulator. `|r|` leaving 1 there is shot noise. The sentence on the Lab plot stays: on a noiseless simulator, deviations from `|r| = 1` are sampling noise, not decoherence. The word coherence is reserved for a real QPU. `fake_fez` would be labelled emulated device noise. No such bank is on disk yet.

The grains in Exposures are samples of the exact `|ψ|²` at `t = T/4`. There is no section 09, and no Atlas shot record over the 256 positions, so they are not Moth Atlas shots.

The echo's delays come from a chain of qubits, not from the proton. The measurement grain in the Experience, including the film flicker at 64, is an illustration of sampling.

## Notes for the Moth team

The tomography schema is the piece that blocks a hardware caption. `provider_name` and `backend_name` are free strings. The only values the document names are `aer` and `automatic`. Other engines list `ibm_fez` and `fake_fez`. Tomography does not, so this project does not invent a device name for it.

When a QPU provider is available, the command above writes a sibling bank. The pages already plot every `src/atlas_bank_*.json` next to the ideal curve and the Aer run. A result is labelled coherence only when the run is classified as hardware (`qpu_job_id` present, and the provider is not `aer`). An emulated backend whose name starts with `fake_` is labelled emulated device noise.

Aer `|r|` at 4096 shots should stay described as shot noise. Please don't read those deviations as decoherence.

Section 09, an 8-qubit encoding with shot counts on the 256 grid sites, was not part of this notebook. Exposures say so, and they sample `|ψ|²` directly. A real position record would switch the caption to "developed from Moth Atlas shots" only when that grid exists.

## Pages

The static site is published from the `docs/` folder:

https://tammy357.github.io/listening-to-a-proton-tunnel/

It holds the Experience, the dark notebook, the EP with players and liner notes, and the contact sheet. The repository is https://github.com/tammy357/listening-to-a-proton-tunnel.

The guided tour for a general audience opens with `?guided=1`. That is the FQxI entry:

https://tammy357.github.io/listening-to-a-proton-tunnel/experience.html?guided=1

## Layout

| Path | Role |
| --- | --- |
| `app.py` | Streamlit bench. Experience, then Lab |
| `src/hamiltonian.py` | Grid, masses, sparse H = T + V |
| `src/quantum_sim.py` | Exact exponential and the split-operator circuit |
| `src/rates.py` | Qualitative Arrhenius sketch, k_tunnel, H/D ratio |
| `src/moth_sonification.py` | Tunneling beat and QPAM measured-state audio |
| `src/experience.html` | Canvas and Web Audio player |
| `src/atlas_tomography.py` | `tomography-api-v2`, with `--backend` |
| `src/atlas_bank.json` | The Aer simulator bank. Not overwritten |
| `src/exposures.py` | Contact sheet and developing films |
| `exposures/` | `contact_sheet.png` and `developing_H.mp4`, `_D`, `_T` |
| `src/atlas_music.py` | Round trip, QRC, Blur, echo, and the EP split |
| `notebooks/media/through_the_wall/` | EP audio, cover, liner notes |
| `src/atlas_tessa.py` | Bloch colour and the carpet |
| `docs/` | GitHub Pages site |
| `notebooks/moth_tunneling_workflow.ipynb` | The workflow, cache or live |
