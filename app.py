"""Streamlit bench for proton tunneling in carbonic anhydrase II.

The page loads the 1HED zinc–water distance, evolves an 8-qubit Gaussian
packet in the asymmetric double well, and draws both the final density and
the donor/acceptor populations. Audio is on demand: the QPAM button measures
the final statevector with Moth Quantum's quantumaudio package and plays the
reconstructed waveform. An Atlas card shows the job document that was posted,
including statevector fidelity, circuit depth, and the active backend.
"""

from __future__ import annotations

import html

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.hamiltonian import ANGSTROM_M, JOULE_PER_EV, WELL_HALF_WIDTH_M
from src.moth_sonification import sonify_tunneling_state
from src.pdb_parser import fetch_active_site
from src.quantum_sim import evolve_tunneling

FEMTOSECOND_S = 1.0e-15

DEFAULT_BARRIER_EV = 0.20
DEFAULT_ASYMMETRY_EV = 0.0
DEFAULT_DURATION_FS = 50.0
DEFAULT_PITCH_HZ = 220.0

INK = "#f3eee6"
MUTED = "#a39888"
AMBER = "#e4a23a"
TEAL = "#3cbaa6"
LINE = "#3a342c"
PANEL = "#191714"
PAPER = "#12110f"


def main() -> None:
    st.set_page_config(
        page_title="Quantum Tunneling Sonification",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    site = _load_site()
    barrier_ev, asymmetry_ev, duration_fs, pitch_hz = _sidebar()
    _header(site)

    try:
        with st.spinner("Evolving the 8-qubit proton statevector…"):
            sim = _simulate(barrier_ev, asymmetry_ev, duration_fs)
    except Exception as exc:
        st.error(f"The tunneling evolution failed: {exc}")
        return

    _population_metrics(sim)
    st.plotly_chart(
        _figure(sim),
        use_container_width=True,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    half_width_angstrom = WELL_HALF_WIDTH_M / ANGSTROM_M
    st.caption(
        f"Well minima sit at ±{half_width_angstrom:.2f} Å. "
        "Positive asymmetry deepens the donor well on the zinc side (x < 0) and "
        "raises the acceptor, so the proton is less willing to tunnel. "
        "The potential axis is clipped to the barrier; the walls keep rising "
        "toward the edges of the 3 Å grid. "
        "Internal arithmetic stays in metres, kilograms, seconds, and joules."
    )
    _audio_section(sim, site, barrier_ev, asymmetry_ev, duration_fs, pitch_hz)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_site() -> dict:
    site = fetch_active_site()
    return {
        "pdb_id": site.pdb_id,
        "distance_angstrom": site.distance_angstrom,
        "source": site.source,
        "note": site.note,
        "zinc_xyz": site.zinc_xyz,
        "water_xyz": site.water_xyz,
    }


@st.cache_data(show_spinner=False)
def _simulate(barrier_ev: float, asymmetry_ev: float, duration_fs: float) -> dict:
    model, evolution = evolve_tunneling(
        barrier_ev=barrier_ev,
        asymmetry_ev=asymmetry_ev,
        duration_s=duration_fs * FEMTOSECOND_S,
    )
    spacing_angstrom = model.spacing_m / ANGSTROM_M
    return {
        "x_angstrom": model.x_m / ANGSTROM_M,
        "v_ev": model.v_joule / JOULE_PER_EV,
        "times_fs": evolution.times_s / FEMTOSECOND_S,
        "donor": evolution.donor_population,
        "acceptor": evolution.acceptor_population,
        "density_per_angstrom": evolution.final_density / spacing_angstrom,
        "initial_density_per_angstrom": (
            np.abs(evolution.initial_statevector) ** 2
        )
        / spacing_angstrom,
        "final_state": evolution.final_statevector,
        "survival": evolution.survival_probability,
    }


def _sidebar() -> tuple[float, float, float, float]:
    st.sidebar.markdown("### Double well")
    st.sidebar.caption("Laboratory scales. The solver converts these to joules and seconds.")
    barrier_ev = st.sidebar.slider(
        "Barrier height (eV)",
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_BARRIER_EV,
        step=0.01,
        help="Height of V(x) at x = 0, between the donor and acceptor wells.",
    )
    asymmetry_ev = st.sidebar.slider(
        "Asymmetry drive (eV)",
        min_value=0.0,
        max_value=0.2,
        value=DEFAULT_ASYMMETRY_EV,
        step=0.005,
        help="λ in V(x) = E_b (1 − (x/a)²)² + λ (x/a). Positive λ biases the proton toward the donor.",
    )
    duration_fs = st.sidebar.slider(
        "Simulation duration (fs)",
        min_value=5.0,
        max_value=50.0,
        value=DEFAULT_DURATION_FS,
        step=1.0,
        help="How long the 8-qubit statevector is propagated.",
    )
    st.sidebar.markdown("### Sonification")
    pitch_hz = st.sidebar.slider(
        "Base audio pitch (Hz)",
        min_value=80.0,
        max_value=880.0,
        value=DEFAULT_PITCH_HZ,
        step=1.0,
        help="The decoded spatial frame is one oscillator cycle. 220 Hz is A3.",
    )
    st.sidebar.caption(
        "8 qubits · 256 sites · x from −1.5 Å to +1.5 Å · QPAM via quantumaudio."
    )
    return (
        float(barrier_ev),
        float(asymmetry_ev),
        float(duration_fs),
        float(pitch_hz),
    )


def _header(site: dict) -> None:
    st.markdown(
        """
        <p class="eyebrow">Moth Hacks 2026 · Human carbonic anhydrase II</p>
        <h1>⚡ Quantum Tunneling &amp; Sonification Engine (Moth Hacks 2026)</h1>
        <p class="lede">A proton on the catalytic water of PDB 1HED sits in a double well
        between the zinc donor and the acceptor side of the active site. This bench
        evolves that proton as an 8-qubit statevector and plays the final amplitudes
        through Moth Quantum's QPAM encoder.</p>
        """,
        unsafe_allow_html=True,
    )
    if site["source"] != "rcsb":
        st.warning(site["note"])
    zinc = ", ".join(f"{value:.3f}" for value in site["zinc_xyz"])
    water = ", ".join(f"{value:.3f}" for value in site["water_xyz"])
    source = "RCSB file" if site["source"] == "rcsb" else "Offline fallback"
    cards = [
        ("PDB ID", site["pdb_id"], "Human carbonic anhydrase II"),
        ("Zn–HOH distance", f"{site['distance_angstrom']:.2f} Å", "ZN A 262 to HOH A 342"),
        ("Coordinate source", source, "Angstroms, orthogonal PDB frame"),
        ("Register", "8 qubits", "256 sites from −1.5 to +1.5 Å"),
    ]
    cells = []
    for label, value, detail in cards:
        cells.append(
            "<div class='metric-card'>"
            f"<span>{html.escape(label)}</span>"
            f"<strong>{html.escape(str(value))}</strong>"
            f"<em>{html.escape(detail)}</em>"
            "</div>"
        )
    st.markdown(f"<div class='metric-grid'>{''.join(cells)}</div>", unsafe_allow_html=True)
    st.caption(f"ZN A 262 at ({zinc}) Å. HOH A 342 at ({water}) Å.")


def _population_metrics(sim: dict) -> None:
    donor = float(sim["donor"][-1])
    acceptor = float(sim["acceptor"][-1])
    survival = float(sim["survival"])
    cards = [
        ("Donor at final time", f"{donor * 100:.1f}%", "Probability on x < 0"),
        ("Acceptor at final time", f"{acceptor * 100:.1f}%", "Probability on x > 0"),
        ("State survival", f"{survival:.3f}", "|⟨ψ(0)|ψ(t)⟩|²"),
        ("Grid", "256 sites", "One amplitude per basis state"),
    ]
    cells = []
    for label, value, detail in cards:
        cells.append(
            "<div class='metric-card metric-card-slim'>"
            f"<span>{html.escape(label)}</span>"
            f"<strong>{html.escape(value)}</strong>"
            f"<em>{html.escape(detail)}</em>"
            "</div>"
        )
    st.markdown(f"<div class='metric-grid'>{''.join(cells)}</div>", unsafe_allow_html=True)


def _potential_window(x_angstrom: np.ndarray, v_ev: np.ndarray) -> list[float]:
    """Y-limits that show the barrier instead of the tall grid-edge walls."""

    x_angstrom = np.asarray(x_angstrom)
    v_ev = np.asarray(v_ev)
    crest = v_ev[np.abs(x_angstrom) <= 0.12]
    top = float(crest.max()) if crest.size else float(np.median(v_ev))
    floor = float(v_ev.min())
    return [floor - 0.08, max(top * 1.45, floor + 0.45)]


def _figure(sim: dict) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.16,
        subplot_titles=(
            "Double-well potential with the final proton density",
            "Donor-well decay and acceptor-well tunneling",
        ),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    figure.add_trace(
        go.Scatter(
            x=sim["x_angstrom"],
            y=sim["v_ev"],
            name="V(x)",
            line={"color": MUTED, "width": 2},
            hovertemplate="x = %{x:.3f} Å<br>V = %{y:.3f} eV<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=sim["x_angstrom"],
            y=sim["initial_density_per_angstrom"],
            name="|ψ(x, 0)|²",
            line={"color": AMBER, "width": 1.5, "dash": "dot"},
            hovertemplate="x = %{x:.3f} Å<br>|ψ|² = %{y:.3f} Å⁻¹<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=sim["x_angstrom"],
            y=sim["density_per_angstrom"],
            name="|ψ(x, t)|²",
            line={"color": TEAL, "width": 2.4},
            fill="tozeroy",
            fillcolor="rgba(60, 186, 166, 0.18)",
            hovertemplate="x = %{x:.3f} Å<br>|ψ|² = %{y:.3f} Å⁻¹<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=sim["times_fs"],
            y=sim["donor"],
            name="Donor",
            line={"color": AMBER, "width": 2.4},
            hovertemplate="t = %{x:.2f} fs<br>P_donor = %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=sim["times_fs"],
            y=sim["acceptor"],
            name="Acceptor",
            line={"color": TEAL, "width": 2.4},
            hovertemplate="t = %{x:.2f} fs<br>P_acceptor = %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.update_xaxes(title_text="x (Å)", row=1, col=1)
    figure.update_xaxes(title_text="t (fs)", row=2, col=1)
    figure.update_yaxes(
        title_text="V(x) (eV)",
        range=_potential_window(sim["x_angstrom"], sim["v_ev"]),
        row=1,
        col=1,
        secondary_y=False,
    )
    figure.update_yaxes(title_text="|ψ|² (Å⁻¹)", row=1, col=1, secondary_y=True)
    figure.update_yaxes(title_text="Population", range=[0, 1.02], row=2, col=1)
    figure.update_layout(
        height=740,
        paper_bgcolor=PAPER,
        plot_bgcolor=PANEL,
        font={"color": INK, "family": "Outfit, Helvetica, sans-serif", "size": 13},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={"l": 60, "r": 50, "t": 70, "b": 50},
        hovermode="x unified",
    )
    figure.update_annotations(font={"color": INK, "size": 14})
    axis_style = {
        "gridcolor": "rgba(243, 238, 230, 0.08)",
        "zerolinecolor": "rgba(243, 238, 230, 0.18)",
        "linecolor": LINE,
    }
    figure.update_xaxes(**axis_style)
    figure.update_yaxes(**axis_style)
    return figure


def _audio_section(
    sim: dict,
    site: dict,
    barrier_ev: float,
    asymmetry_ev: float,
    duration_fs: float,
    pitch_hz: float,
) -> None:
    token = (
        round(barrier_ev, 4),
        round(asymmetry_ev, 4),
        round(duration_fs, 2),
        round(pitch_hz, 1),
    )
    if st.session_state.get("audio_token") != token:
        st.session_state.pop("sonification", None)
        st.session_state["audio_token"] = token

    st.markdown("### Moth Quantum audio")
    column_copy, column_play = st.columns([1.15, 1], gap="large")
    with column_copy:
        st.markdown(
            "The final statevector is phase-aligned, split into two real "
            "quadratures, and encoded with "
            "`quantumaudio.encode(..., scheme=\"qpam\")`. "
            "Decoding measures that circuit. The reconstructed frame is one "
            "cycle of the oscillator, so the pitch slider sets how fast the "
            "proton's spatial amplitude is scanned. Loudness follows the "
            "acceptor population: tunneling opens the tone."
        )
        synthesize = st.button(
            "🎵 Synthesize Quantum Audio (QPAM)",
            type="primary",
            use_container_width=True,
        )
    with column_play:
        if synthesize:
            with st.spinner("Measuring the QPAM circuit and filing the Atlas job…"):
                try:
                    st.session_state["sonification"] = sonify_tunneling_state(
                        sim["final_state"],
                        base_pitch=pitch_hz,
                        envelope=sim["acceptor"],
                        atlas_context={
                            "pdb_id": site["pdb_id"],
                            "zn_hoh_distance_angstrom": round(
                                float(site["distance_angstrom"]), 4
                            ),
                            "coordinate_source": site["source"],
                            "barrier_ev": barrier_ev,
                            "asymmetry_ev": asymmetry_ev,
                            "duration_fs": duration_fs,
                            "base_pitch_hz": pitch_hz,
                            "donor_population_final": round(float(sim["donor"][-1]), 6),
                            "acceptor_population_final": round(
                                float(sim["acceptor"][-1]), 6
                            ),
                            "initial_state_overlap": round(float(sim["survival"]), 6),
                        },
                    )
                except Exception as exc:
                    st.session_state.pop("sonification", None)
                    st.error(f"QPAM synthesis failed: {exc}")

        result = st.session_state.get("sonification")
        if result is None:
            st.markdown(
                """
                <div class="quiet">
                  <strong>No measurement yet</strong>
                  <p>The plots are already live. Synthesize to hear this statevector
                  after QPAM encoding and shot reconstruction.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.audio(result.wav_bytes, format="audio/wav")
            st.plotly_chart(
                _waveform_figure(result.waveform, result.sample_rate),
                use_container_width=True,
                config={"displaylogo": False, "displayModeBar": False},
            )

    result = st.session_state.get("sonification")
    if result is not None:
        dispatched = bool(result.atlas_job.get("atlas", {}).get("dispatched"))
        status = (
            "Atlas accepted the posted job."
            if dispatched
            else "Fidelity, depth, and backend come from the local Aer measurement. "
            "The atlas block is the platform's actual HTTP response."
        )
        st.caption(status)
        with st.expander("🌐 Moth Atlas Platform API Output", expanded=True):
            st.json(result.atlas_job)


def _waveform_figure(samples: np.ndarray, sample_rate: int) -> go.Figure:
    stride = max(len(samples) // 1400, 1)
    taken = np.asarray(samples)[::stride]
    times_ms = np.arange(taken.size) * stride / sample_rate * 1000.0
    figure = go.Figure(
        go.Scatter(
            x=times_ms,
            y=taken,
            mode="lines",
            line={"color": AMBER, "width": 1.4},
            hovertemplate="t = %{x:.1f} ms<br>amplitude = %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=220,
        paper_bgcolor=PAPER,
        plot_bgcolor=PANEL,
        font={"color": INK, "family": "Outfit, Helvetica, sans-serif", "size": 12},
        margin={"l": 48, "r": 16, "t": 28, "b": 40},
        title={"text": "Played waveform", "font": {"size": 14, "color": INK}},
        xaxis={"title": "t (ms)", "gridcolor": "rgba(243,238,230,0.08)", "linecolor": LINE},
        yaxis={
            "title": "Amplitude",
            "range": [-1.05, 1.05],
            "gridcolor": "rgba(243,238,230,0.08)",
            "linecolor": LINE,
        },
    )
    return figure


def _inject_css() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,520;9..144,640&family=Outfit:wght@380;520;600&display=swap" rel="stylesheet">
        <style>
          html, body, [class*="css"] {
            font-family: "Outfit", Helvetica, sans-serif;
          }
          .block-container { padding-top: 1.3rem; max-width: 1180px; }
          h1 {
            font-family: "Fraunces", Georgia, serif !important;
            font-weight: 560 !important;
            letter-spacing: -0.03em;
            font-size: 2.15rem !important;
            line-height: 1.15 !important;
            margin-bottom: 0.35rem;
          }
          .eyebrow {
            color: #e4a23a;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-size: 0.74rem;
            margin: 0 0 0.35rem 0;
          }
          .lede { color: #c9c0b2; font-size: 1.05rem; max-width: 46rem; margin-bottom: 0.4rem; }
          .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.8rem 0 1rem;
          }
          .metric-card {
            background: #1c1a16;
            border: 1px solid #3a342c;
            border-radius: 14px;
            padding: 0.8rem 0.95rem 0.85rem;
          }
          .metric-card span {
            display: block;
            color: #a39888;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          .metric-card strong {
            display: block;
            font-family: "Fraunces", Georgia, serif;
            font-size: 1.45rem;
            font-weight: 560;
            margin-top: 0.18rem;
            color: #f3eee6;
          }
          .metric-card em {
            display: block;
            color: #a39888;
            font-style: normal;
            font-size: 0.8rem;
            margin-top: 0.15rem;
          }
          .quiet {
            border: 1px dashed #3a342c;
            border-radius: 14px;
            padding: 1.1rem 1rem;
            background: #191714;
            min-height: 7.5rem;
          }
          .quiet strong { display: block; font-size: 1.02rem; margin-bottom: 0.3rem; }
          .quiet p { color: #a39888; margin: 0; }
          div[data-testid="stExpander"] {
            border: 1px solid #3a342c;
            border-radius: 14px;
            background: #1c1a16;
          }
          @media (max-width: 860px) {
            .metric-grid { grid-template-columns: 1fr 1fr; }
            h1 { font-size: 1.65rem !important; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


main()
